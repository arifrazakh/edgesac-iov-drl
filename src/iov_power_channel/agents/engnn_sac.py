
from __future__ import annotations

import csv
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Batch, Data
from torch_geometric.nn import MetaLayer
from torch_scatter import scatter

try:
    from torch.amp import autocast, GradScaler as TorchGradScaler
    _AMP_HAS_DEVICE_ARG = True
except Exception:
    from torch.cuda.amp import autocast, GradScaler as TorchGradScaler
    _AMP_HAS_DEVICE_ARG = False

from iov_power_channel.envs.mobile_network_env import OBS_IDX, MobileNetwork


def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)
    return layer


class ReplayBuffer:
    def __init__(self, num_nodes: int, node_feature_dim: int, act_dim: int, size: int, batch_size: int = 256):
        self.node_features_buf = np.zeros([size, num_nodes, node_feature_dim], dtype=np.float32)
        self.next_node_features_buf = np.zeros([size, num_nodes, node_feature_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.term_buf = np.zeros([size], dtype=np.float32)
        self.timeout_buf = np.zeros([size], dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size = 0, 0
        self.num_nodes = num_nodes
        self.node_feature_dim = node_feature_dim
        self.action_dim = act_dim

    def store(self, node_features: np.ndarray, act: np.ndarray, rew: float,
              next_node_features: np.ndarray, term: bool, timeout: bool):
        self.node_features_buf[self.ptr] = node_features
        self.next_node_features_buf[self.ptr] = next_node_features
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.term_buf[self.ptr] = float(term)
        self.timeout_buf[self.ptr] = float(timeout)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> dict:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(
            node_features=self.node_features_buf[idxs],
            next_node_features=self.next_node_features_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            term=self.term_buf[idxs],
            timeout=self.timeout_buf[idxs],
        )

    def __len__(self) -> int:
        return self.size


class _EdgeModel(nn.Module):
    def __init__(self, node_in, edge_in, hid=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_in * 2 + edge_in, hid), nn.ReLU(),
            nn.Linear(hid, edge_in)
        )

    def forward(self, src, dst, edge_attr, u, batch):
        return edge_attr + self.mlp(torch.cat([src, dst, edge_attr], dim=-1))


class _NodeModel(nn.Module):
    def __init__(self, node_in, edge_in, hid=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_in + edge_in, hid), nn.ReLU(),
            nn.Linear(hid, node_in)
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        agg = scatter(edge_attr, col, dim=0, reduce="mean", dim_size=x.size(0))
        return x + self.mlp(torch.cat([x, agg], dim=-1))


class _ENGNNBlock(nn.Module):
    def __init__(self, node_dim, edge_dim):
        super().__init__()
        self.layer = MetaLayer(_EdgeModel(node_dim, edge_dim), _NodeModel(node_dim, edge_dim), None)
        self.nx = nn.LayerNorm(node_dim)
        self.ne = nn.LayerNorm(edge_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        x2, e2, _ = self.layer(x, edge_index, edge_attr, u=None, batch=batch)
        return self.nx(x2), self.ne(e2)


class ENGNNActor(nn.Module):
    def __init__(self, node_in, edge_in, out_dim_per_node, blocks=2, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.x_proj = nn.Linear(node_in, 64)
        self.e_proj = nn.Linear(edge_in, 32)
        self.blocks = nn.ModuleList([_ENGNNBlock(64, 32) for _ in range(blocks)])
        self.mu = init_layer_uniform(nn.Linear(64, out_dim_per_node))
        self.ls = init_layer_uniform(nn.Linear(64, out_dim_per_node))
        self.log_std_min, self.log_std_max = log_std_min, log_std_max

    def forward(self, data: Data):
        x, ei, ea, b = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.relu(self.x_proj(x))
        ea = F.relu(self.e_proj(ea))
        for bl in self.blocks:
            x, ea = bl(x, ei, ea, b)
        mu = self.mu(x)
        log_std = self.ls(x).tanh()
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        std = torch.exp(log_std).clamp(1e-6, 1e6)
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        a = torch.tanh(z.clamp(-10, 10))
        eps = 1e-6
        logp = dist.log_prob(z) - torch.log((1 - a.pow(2)).clamp(min=eps))
        logp = scatter(logp.sum(-1, keepdim=True), b, dim=0, reduce="sum")
        return a, logp


class ENGNNQ(nn.Module):
    def __init__(self, node_in, edge_in, act_dim_per_node, blocks=2):
        super().__init__()
        self.x_proj = nn.Linear(node_in + act_dim_per_node, 64)
        self.e_proj = nn.Linear(edge_in, 32)
        self.blocks = nn.ModuleList([_ENGNNBlock(64, 32) for _ in range(blocks)])
        self.out = init_layer_uniform(nn.Linear(64, 1))

    def forward(self, data: Data, action: torch.Tensor):
        x, ei, ea, b = data.x, data.edge_index, data.edge_attr, data.batch
        if action.dim() == 3:
            action = action.view(-1, action.size(-1))
        x = torch.cat([x, action], dim=-1)
        x = F.relu(self.x_proj(x))
        ea = F.relu(self.e_proj(ea))
        for bl in self.blocks:
            x, ea = bl(x, ei, ea, b)
        v = self.out(x)
        return scatter(v, b, dim=0, reduce="mean")


class ENGNNV(nn.Module):
    def __init__(self, node_in, edge_in, blocks=2):
        super().__init__()
        self.x_proj = nn.Linear(node_in, 64)
        self.e_proj = nn.Linear(edge_in, 32)
        self.blocks = nn.ModuleList([_ENGNNBlock(64, 32) for _ in range(blocks)])
        self.out = init_layer_uniform(nn.Linear(64, 1))

    def forward(self, data: Data):
        x, ei, ea, b = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.relu(self.x_proj(x))
        ea = F.relu(self.e_proj(ea))
        for bl in self.blocks:
            x, ea = bl(x, ei, ea, b)
        v = self.out(x)
        return scatter(v, b, dim=0, reduce="mean")


class SACAgent:
    def __init__(
        self,
        env: MobileNetwork,
        bs_loc,
        memory_size: int = int(1e5),
        batch_size: int = 256,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        initial_random_steps: int = 1000,
        policy_update_freq: int = 2,
        seed: int = 42,
        n_neighbors: int = 6,
    ):
        self.env = env
        self.bs_loc = np.array(bs_loc, dtype=np.float32)
        obs_shape = env.observation_space.shape
        self.num_nodes = obs_shape[0]
        self.raw_node_feature_dim = obs_shape[1]
        self.node_feature_dim = self.raw_node_feature_dim + 2
        self.edge_feat_dim = 6
        self.action_dim_per_node = env.action_space.shape[1]
        self.action_dim = self.num_nodes * self.action_dim_per_node
        self.action_shape = env.action_space.shape
        self.n_neighbors = max(2, int(n_neighbors))

        self.memory = ReplayBuffer(self.num_nodes, self.raw_node_feature_dim, self.action_dim, memory_size, batch_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.initial_random_steps = initial_random_steps
        self.policy_update_freq = policy_update_freq
        self.seed = seed

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = self.device.type == "cuda"
        self.amp_dtype = (
            torch.bfloat16 if (self.device.type == "cuda" and torch.cuda.is_bf16_supported())
            else torch.float16
        )
        if self.use_amp and _AMP_HAS_DEVICE_ARG:
            self.scaler = TorchGradScaler(device="cuda", enabled=True)
        else:
            self.scaler = TorchGradScaler(enabled=self.use_amp)

        def _amp_cm(enabled=True, dtype=self.amp_dtype):
            if not enabled:
                return nullcontext()
            if _AMP_HAS_DEVICE_ARG:
                return autocast(device_type="cuda", dtype=dtype, enabled=True)
            try:
                return autocast(enabled=True, dtype=dtype)
            except TypeError:
                return autocast(enabled=True)
        self._amp_cm = _amp_cm

        self.auto_alpha = True
        init_alpha = 0.2
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(init_alpha, device=self.device)), requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)
        self.target_entropy = -0.10 * (self.num_nodes * self.action_dim_per_node)

        self.actor = ENGNNActor(self.node_feature_dim, self.edge_feat_dim, self.action_dim_per_node, blocks=2).to(self.device)
        self.vf = ENGNNV(self.node_feature_dim, self.edge_feat_dim, blocks=2).to(self.device)
        self.vf_target = ENGNNV(self.node_feature_dim, self.edge_feat_dim, blocks=2).to(self.device)
        self.vf_target.load_state_dict(self.vf.state_dict())
        self.vf_target.eval()
        for p in self.vf_target.parameters():
            p.requires_grad = False

        self.qf_1 = ENGNNQ(self.node_feature_dim, self.edge_feat_dim, self.action_dim_per_node, blocks=2).to(self.device)
        self.qf_2 = ENGNNQ(self.node_feature_dim, self.edge_feat_dim, self.action_dim_per_node, blocks=2).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.vf_optimizer = optim.Adam(self.vf.parameters(), lr=learning_rate)
        self.qf_1_optimizer = optim.Adam(self.qf_1.parameters(), lr=learning_rate)
        self.qf_2_optimizer = optim.Adam(self.qf_2.parameters(), lr=learning_rate)

        self.transition = []
        self.total_step = 0
        self.is_test = False
        self.episode_rewards: List[float] = []
        self.training_rows: List[Dict[str, float]] = []

        self.static_edge_index = self._compute_edge_index()
        one_hot = np.zeros((self.num_nodes, 2), dtype=np.float32)
        for i, bs in enumerate(self.env.base_stations):
            one_hot[i] = [1, 0] if bs.type_bs == "Ma" else [0, 1]
        self.type_features = torch.from_numpy(one_hot).to(self.device)

    def _compute_edge_index(self):
        n = self.num_nodes
        if n <= 1:
            edge_list = [[0, 0]]
        elif n == 2:
            edge_list = [[0, 1], [1, 0]]
        else:
            n_nbrs = min(self.n_neighbors, n)
            nbrs = NearestNeighbors(n_neighbors=n_nbrs, algorithm="ball_tree").fit(self.bs_loc)
            _, indices = nbrs.kneighbors(self.bs_loc)
            edge_list = []
            for i in range(n):
                for j in indices[i][1:]:
                    edge_list.append([i, j]); edge_list.append([j, i])
            if not edge_list:
                for i in range(n):
                    j = (i + 1) % n
                    edge_list.append([i, j]); edge_list.append([j, i])
        return torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(self.device)

    def _edge_features(self, obs: np.ndarray) -> torch.Tensor:
        tx_norm = torch.as_tensor(obs[:, OBS_IDX.TX_NORM], dtype=torch.float32, device=self.device)
        inter_norm = torch.as_tensor(obs[:, OBS_IDX.INTER_NORM], dtype=torch.float32, device=self.device)
        types01 = self.type_features
        freqs = torch.as_tensor([self.env.bs_carrier_frequency[i] for i in range(self.num_nodes)], dtype=torch.float32, device=self.device)
        xy = torch.as_tensor(self.bs_loc, dtype=torch.float32, device=self.device)
        src, dst = self.static_edge_index
        d = (xy[src] - xy[dst]).pow(2).sum(-1).sqrt()
        d_norm = (d / 1000.0).clamp(0, 1)
        same_f = (freqs[src] == freqs[dst]).float()
        pair_ma = (types01[src, 0] * types01[dst, 0]).float()
        pair_mi = (types01[src, 1] * types01[dst, 1]).float()
        tx_diff = (tx_norm[src] - tx_norm[dst]).abs()
        inter_avg = 0.5 * (inter_norm[src] + inter_norm[dst])
        return torch.stack([d_norm, same_f, pair_ma, pair_mi, tx_diff, inter_avg], dim=-1)

    def create_graph_data(self, obs, add_batch: bool = False):
        obs = np.nan_to_num(obs, nan=0.0)
        x_obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        x = torch.cat([x_obs, self.type_features], dim=1)
        data = Data(x=x, edge_index=self.static_edge_index, edge_attr=self._edge_features(obs))
        if add_batch:
            data.batch = torch.zeros(self.num_nodes, dtype=torch.long, device=self.device)
        return data

    @staticmethod
    def _scale_env_action(a_raw: torch.Tensor) -> torch.Tensor:
        low, high = 0.001, 1.0
        return low + (high - low) * (a_raw + 1.0) / 2.0

    def _apply_action_floor_tensor(self, a_env: torch.Tensor, data):
        req_p_norm = data.x[:, OBS_IDX.REQ_P_NORM:OBS_IDX.REQ_P_NORM + 1].clamp(0, 1)
        cov_util = data.x[:, OBS_IDX.COV_UTIL:OBS_IDX.COV_UTIL + 1].clamp(0, 1)
        inter_norm = data.x[:, OBS_IDX.INTER_NORM:OBS_IDX.INTER_NORM + 1].clamp(0, 1)

        p_floor = torch.maximum(torch.full_like(req_p_norm, 0.05), 0.8 * req_p_norm)
        p = torch.maximum(a_env[:, 0:1], p_floor)
        ch_floor = (0.05 + 0.6 * cov_util).clamp(0.05, 0.8)
        ch_upper = (1.0 - 0.5 * inter_norm).clamp(0.3, 1.0)
        ch = torch.minimum(torch.maximum(a_env[:, 1:2], ch_floor), ch_upper)
        return torch.cat([p, ch], dim=-1)

    def select_action(self, state: np.ndarray) -> np.ndarray:
        if self.total_step < self.initial_random_steps and not self.is_test:
            action = self.env.action_space.sample()
            action_flat = action.flatten()
        else:
            was_training = self.actor.training
            self.actor.eval()
            with torch.no_grad():
                data = self.create_graph_data(state, add_batch=True)
                a_raw, _ = self.actor(data)
                a_env = self._scale_env_action(a_raw)
                a_env = self._apply_action_floor_tensor(a_env, data)
                action_flat = a_env.detach().cpu().numpy().reshape(-1)
            if was_training and not self.is_test:
                self.actor.train()
        self.transition = [state, action_flat]
        return action_flat.reshape(self.action_shape)

    def env_step(self, action: np.ndarray):
        next_state, env_reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        reward_scaled = env_reward / 10.0
        if not self.is_test:
            action_flat = action.flatten()
            self.transition += [reward_scaled, next_state, done]
            self.memory.store(self.transition[0], action_flat, reward_scaled, next_state, terminated, truncated)
        return next_state, env_reward, done, info

    def update_model(self):
        samples = self.memory.sample_batch()
        device = self.device

        action = torch.FloatTensor(samples["acts"]).to(device).view(-1, self.num_nodes, self.action_dim_per_node)
        reward = torch.FloatTensor(samples["rews"]).to(device).reshape(-1, 1)
        term = torch.FloatTensor(samples["term"]).to(device).reshape(-1, 1)
        not_terminal = 1.0 - term

        data_list = [self.create_graph_data(obs, add_batch=False) for obs in samples["node_features"]]
        next_data_list = [self.create_graph_data(obs, add_batch=False) for obs in samples["next_node_features"]]
        batch = Batch.from_data_list(data_list).to(device)
        next_batch = Batch.from_data_list(next_data_list).to(device)

        with self._amp_cm(enabled=self.use_amp):
            with torch.no_grad():
                v_target = self.vf_target(next_batch)
                q_target = reward + self.gamma * v_target * not_terminal
                q_target = q_target.clamp(-50.0, 50.0)

        with self._amp_cm(enabled=self.use_amp):
            q1_pred = self.qf_1(batch, action)
            q2_pred = self.qf_2(batch, action)
            qf_loss = F.smooth_l1_loss(q1_pred, q_target) + F.smooth_l1_loss(q2_pred, q_target)

        self.qf_1_optimizer.zero_grad()
        self.qf_2_optimizer.zero_grad()
        self.scaler.scale(qf_loss).backward()
        self.scaler.unscale_(self.qf_1_optimizer)
        self.scaler.unscale_(self.qf_2_optimizer)
        torch.nn.utils.clip_grad_norm_(self.qf_1.parameters(), 5.0)
        torch.nn.utils.clip_grad_norm_(self.qf_2.parameters(), 5.0)
        self.scaler.step(self.qf_1_optimizer)
        self.scaler.step(self.qf_2_optimizer)

        actor_loss = torch.tensor(0.0, device=device)
        vf_loss = torch.tensor(0.0, device=device)
        alpha_loss = torch.tensor(0.0, device=device)

        if self.total_step % max(1, self.policy_update_freq) == 0:
            with self._amp_cm(enabled=self.use_amp):
                new_action_raw, log_prob = self.actor(batch)
                log_prob = torch.nan_to_num(log_prob, nan=0.0, posinf=0.0, neginf=0.0)
                new_action_env = self._scale_env_action(new_action_raw)
                new_action_env = self._apply_action_floor_tensor(new_action_env, batch)
                alpha_const = self.log_alpha.exp().detach()
                q_pi = torch.min(self.qf_1(batch, new_action_env), self.qf_2(batch, new_action_env))
                actor_loss = (alpha_const * log_prob - q_pi).mean()

            self.actor_optimizer.zero_grad()
            self.scaler.scale(actor_loss).backward()
            self.scaler.unscale_(self.actor_optimizer)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
            self.scaler.step(self.actor_optimizer)

            if self.auto_alpha:
                with self._amp_cm(enabled=self.use_amp):
                    alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                self.scaler.scale(alpha_loss).backward()
                self.scaler.step(self.alpha_optimizer)
                with torch.no_grad():
                    self.log_alpha.clamp_(min=np.log(1e-4), max=np.log(10.0))

            with self._amp_cm(enabled=self.use_amp):
                v_pred = self.vf(batch)
                with torch.no_grad():
                    na_raw, lp = self.actor(batch)
                    na_env = self._scale_env_action(na_raw)
                    na_env = self._apply_action_floor_tensor(na_env, batch)
                    q_pi_detached = torch.min(self.qf_1(batch, na_env), self.qf_2(batch, na_env))
                    v_tgt2 = q_pi_detached - self.log_alpha.exp().detach() * lp
                vf_loss = F.mse_loss(v_pred, v_tgt2)

            self.vf_optimizer.zero_grad()
            self.scaler.scale(vf_loss).backward()
            self.scaler.unscale_(self.vf_optimizer)
            torch.nn.utils.clip_grad_norm_(self.vf.parameters(), 5.0)
            self.scaler.step(self.vf_optimizer)

        self.scaler.update()
        self._target_soft_update()
        return float(actor_loss.detach().cpu()), float(qf_loss.detach().cpu()), float(vf_loss.detach().cpu()), float(alpha_loss.detach().cpu())

    def train(self, num_frames: int, log_csv_path: Optional[str] = None):
        self.is_test = False
        state, _ = self.env.reset(seed=self.seed)
        score = 0.0
        ep_steps = 0
        ep_count = 0
        actor_losses, qf_losses, vf_losses = [], [], []

        writer = None
        f = None
        if log_csv_path:
            Path(log_csv_path).parent.mkdir(parents=True, exist_ok=True)
            f = open(log_csv_path, "w", newline="", encoding="utf-8")
            writer = csv.writer(f)
            writer.writerow(["episode", "reward", "steps", "util", "avg_rate", "avg_latency", "total_power", "actor_loss", "qf_loss", "vf_loss"])

        for self.total_step in range(1, num_frames + 1):
            action = self.select_action(state)
            next_state, reward, done, info = self.env_step(action)
            state = next_state
            score += reward
            ep_steps += 1

            if len(self.memory) >= self.batch_size and self.total_step > self.initial_random_steps:
                a, q, v, al = self.update_model()
                actor_losses.append(a)
                qf_losses.append(q)
                vf_losses.append(v)

            if done:
                ep_count += 1
                row = {
                    "episode": ep_count,
                    "reward": float(score),
                    "steps": float(ep_steps),
                    "util": float(info.get("util", 0.0)),
                    "avg_rate": float(info.get("avg_rate", 0.0)),
                    "avg_latency": float(info.get("avg_latency", 0.0)),
                    "total_power": float(info.get("total_power", 0.0)),
                    "actor_loss": float(np.mean(actor_losses)) if actor_losses else np.nan,
                    "qf_loss": float(np.mean(qf_losses)) if qf_losses else np.nan,
                    "vf_loss": float(np.mean(vf_losses)) if vf_losses else np.nan,
                }
                self.training_rows.append(row)
                self.episode_rewards.append(float(score))
                if writer is not None:
                    writer.writerow(list(row.values()))
                score = 0.0
                ep_steps = 0
                actor_losses.clear(); qf_losses.clear(); vf_losses.clear()
                state, _ = self.env.reset()

        if f is not None:
            f.close()
        return self.training_rows

    def test(self, num_episodes: int = 5) -> List[Dict[str, float]]:
        self.is_test = True
        self.actor.eval(); self.vf.eval(); self.qf_1.eval(); self.qf_2.eval()
        rows = []
        for ep in range(1, num_episodes + 1):
            state, _ = self.env.reset(seed=self.seed + ep)
            done = False
            score = 0.0
            infos = []
            while not done:
                action = self.select_action(state)
                next_state, reward, done, info = self.env_step(action)
                state = next_state
                score += reward
                infos.append(info)
            merged = {"episode": ep, "reward": float(score)}
            for k in set().union(*[d.keys() for d in infos]):
                vals = [float(d[k]) for d in infos if k in d]
                if vals:
                    merged[k] = float(np.mean(vals))
            rows.append(merged)
        return rows

    def _target_soft_update(self):
        for t_param, l_param in zip(self.vf_target.parameters(), self.vf.parameters()):
            t_param.data.copy_(self.tau * l_param.data + (1.0 - self.tau) * t_param.data)

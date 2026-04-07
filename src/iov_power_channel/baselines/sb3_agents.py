
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Type

import numpy as np
from stable_baselines3 import A2C, PPO

from iov_power_channel.envs.mobile_network_env import FlattenActionObservationWrapper, MobileNetwork


ALGOS = {
    "PPO": PPO,
    "A2C": A2C,
}


def make_sb3_env(env: MobileNetwork) -> FlattenActionObservationWrapper:
    return FlattenActionObservationWrapper(env)


def train_sb3(
    algo_name: str,
    env: MobileNetwork,
    train_steps: int = 30000,
    seed: int = 42,
):
    algo_cls = ALGOS[algo_name]
    wrapped = make_sb3_env(env)
    model = algo_cls(
        "MlpPolicy",
        wrapped,
        verbose=0,
        seed=seed,
    )
    model.learn(total_timesteps=train_steps, progress_bar=False)
    return model


def evaluate_sb3(model, env: MobileNetwork, num_episodes: int = 5, seed: int = 42) -> List[Dict[str, float]]:
    rows = []
    wrapped = make_sb3_env(env)
    for ep in range(1, num_episodes + 1):
        obs, _ = wrapped.reset(seed=seed + ep)
        done = False
        reward_sum = 0.0
        infos = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = wrapped.step(action)
            done = terminated or truncated
            reward_sum += float(reward)
            infos.append(info)
        row = {"episode": ep, "reward": float(reward_sum)}
        if infos:
            keys = sorted({k for d in infos for k in d.keys()})
            for k in keys:
                vals = [float(d[k]) for d in infos if k in d]
                if vals:
                    row[k] = float(np.mean(vals))
        rows.append(row)
    return rows

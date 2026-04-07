
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from iov_power_channel.agents.engnn_sac import SACAgent
from iov_power_channel.baselines.heuristics import LoadAwareHeuristicPolicy, MaxPowerMaxChannelPolicy, RandomPolicy
from iov_power_channel.baselines.sb3_agents import evaluate_sb3, train_sb3
from iov_power_channel.envs.mobile_network_env import MobileNetwork, default_bs_locations
from iov_power_channel.utils.common import ensure_dir, seed_everything


def make_env(seed: int = 42) -> MobileNetwork:
    return MobileNetwork(
        num_base_stations=36,
        num_users=100,
        num_channels=265,
        area_size=5000,
        bs_loc=default_bs_locations(),
        mobility_model="manhattan",
        road_spacing=250.0,
        v_mean=12.0,
        v_std=3.0,
        stop_prob=0.05,
        max_stop_steps=3,
        max_steps=200,
        render_mode="none",
        seed=seed,
    )


def evaluate_policy(env: MobileNetwork, policy, eval_episodes: int = 5, seed: int = 42):
    rows = []
    for ep in range(1, eval_episodes + 1):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        reward_sum = 0.0
        infos = []
        while not done:
            action = policy.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            reward_sum += float(reward)
            infos.append(info)
        row = {"episode": ep, "reward": float(reward_sum)}
        for k in sorted({k for d in infos for k in d.keys()}):
            vals = [float(d[k]) for d in infos if k in d]
            if vals:
                row[k] = float(np.mean(vals))
        rows.append(row)
    return rows


def add_result_block(rows, algo_name, all_eval_rows, all_summary_rows):
    for r in rows:
        rr = dict(r)
        rr["algorithm"] = algo_name
        all_eval_rows.append(rr)

    df = pd.DataFrame(rows)
    summary = {
        "algorithm": algo_name,
        "reward_mean": float(df["reward"].mean()) if "reward" in df else np.nan,
        "reward_std": float(df["reward"].std(ddof=0)) if "reward" in df else np.nan,
        "util_mean": float(df["util"].mean()) if "util" in df else np.nan,
        "avg_rate_mean": float(df["avg_rate"].mean()) if "avg_rate" in df else np.nan,
        "avg_latency_mean": float(df["avg_latency"].mean()) if "avg_latency" in df else np.nan,
        "total_power_mean": float(df["total_power"].mean()) if "total_power" in df else np.nan,
        "jain_fairness_mean": float(df["jain_fairness"].mean()) if "jain_fairness" in df else np.nan,
        "energy_eff_Mb_per_J_mean": float(df["energy_eff_Mb_per_J"].mean()) if "energy_eff_Mb_per_J" in df else np.nan,
    }
    all_summary_rows.append(summary)


def save_bar_plot(summary_df: pd.DataFrame, out_path: Path):
    if summary_df.empty:
        return
    plt.figure(figsize=(10, 5))
    x = np.arange(len(summary_df))
    y = summary_df["reward_mean"].to_numpy()
    err = summary_df["reward_std"].fillna(0).to_numpy()
    plt.bar(x, y)
    plt.xticks(x, summary_df["algorithm"], rotation=20)
    plt.ylabel("Mean episode reward")
    plt.title("Algorithm comparison")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all", "proposed", "baselines", "heuristics"], default="all")
    parser.add_argument("--train-steps", type=int, default=30000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    seed_everything(args.seed)
    out_dir = ensure_dir(args.output_dir)

    all_eval_rows = []
    all_summary_rows = []

    if args.mode in {"all", "proposed"}:
        env_prop = make_env(seed=args.seed)
        agent = SACAgent(
            env=env_prop,
            bs_loc=default_bs_locations(),
            memory_size=int(1e5),
            batch_size=256,
            initial_random_steps=1000,
            policy_update_freq=2,
            seed=args.seed,
            n_neighbors=6,
        )
        train_csv = out_dir / "proposed_training.csv"
        agent.train(num_frames=args.train_steps, log_csv_path=str(train_csv))
        eval_rows = agent.test(num_episodes=args.eval_episodes)
        add_result_block(eval_rows, "ENGNNSAC", all_eval_rows, all_summary_rows)
        env_prop.close()

    if args.mode in {"all", "baselines"}:
        for algo_name in ["PPO", "A2C"]:
            env_b = make_env(seed=args.seed)
            model = train_sb3(algo_name, env_b, train_steps=args.train_steps, seed=args.seed)
            eval_rows = evaluate_sb3(model, env_b, num_episodes=args.eval_episodes, seed=args.seed)
            add_result_block(eval_rows, algo_name, all_eval_rows, all_summary_rows)
            env_b.close()

    if args.mode in {"all", "heuristics"}:
        for policy_cls in [RandomPolicy, LoadAwareHeuristicPolicy, MaxPowerMaxChannelPolicy]:
            env_h = make_env(seed=args.seed)
            policy = policy_cls(env_h)
            eval_rows = evaluate_policy(env_h, policy, eval_episodes=args.eval_episodes, seed=args.seed)
            add_result_block(eval_rows, policy.name, all_eval_rows, all_summary_rows)
            env_h.close()

    eval_df = pd.DataFrame(all_eval_rows)
    summary_df = pd.DataFrame(all_summary_rows)

    eval_csv = out_dir / "per_episode_metrics.csv"
    summary_csv = out_dir / "comparison_summary.csv"
    eval_df.to_csv(eval_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    save_bar_plot(summary_df, out_dir / "comparison_bar.png")

    print("\nSaved:")
    print(f"- {eval_csv}")
    print(f"- {summary_csv}")
    print(f"- {out_dir / 'comparison_bar.png'}")


if __name__ == "__main__":
    main()

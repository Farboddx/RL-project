from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

from agent import ActorCriticAgent, ReinforceAgent, parse_hidden_sizes, set_seed
from env_utils import assert_continuous_box_space, make_env, reset_env, step_env


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def evaluate(agent, env, episodes: int, seed: int | None = None) -> tuple[float, float]:
    returns: list[float] = []
    for episode in range(episodes):
        eval_seed = None if seed is None else seed + 10_000 + episode
        obs, _ = reset_env(env, seed=eval_seed)
        done = False
        total = 0.0
        while not done:
            action, _ = agent.select_action(obs, deterministic=True)
            obs, reward, done, _ = step_env(env, action)
            total += reward
        returns.append(total)
    return float(np.mean(returns)), float(np.std(returns))


def run_training_episode(agent, env, seed: int | None = None) -> tuple[float, int]:
    obs, _ = reset_env(env, seed=seed)
    done = False
    total = 0.0
    length = 0
    while not done:
        action, action_info = agent.select_action(obs, deterministic=False)
        next_obs, reward, done, _ = step_env(env, action)
        agent.record(obs, action_info, reward, next_obs, done)
        obs = next_obs
        total += reward
        length += 1
    return total, length


def build_agent(args, env):
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    common = {
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "action_low": env.action_space.low,
        "action_high": env.action_space.high,
        "hidden_sizes": hidden_sizes,
        "gamma": args.gamma,
        "entropy_coef": args.entropy_coef,
        "max_grad_norm": args.max_grad_norm,
        "normalize_advantages": args.normalize_advantages,
        "device": args.device,
    }

    if args.algo == "reinforce":
        return ReinforceAgent(
            **common,
            lr=args.actor_lr,
            baseline=0.0,
            reward_to_go=not args.full_trajectory_return,
        )
    if args.algo == "reinforce_baseline":
        return ReinforceAgent(
            **common,
            lr=args.actor_lr,
            baseline=args.baseline,
            reward_to_go=not args.full_trajectory_return,
        )
    if args.algo == "actor_critic":
        return ActorCriticAgent(
            **common,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            value_coef=args.value_coef,
            target_mode=args.ac_target,
        )
    raise ValueError(f"Unknown algorithm: {args.algo}")


def write_args(path: Path, args) -> None:
    payload = vars(args).copy()
    payload["created_at"] = datetime.now().isoformat(timespec="seconds")
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train(args) -> Path:
    set_seed(args.seed)
    train_env, backend = make_env(args.env_id, seed=args.seed, max_episode_steps=args.max_episode_steps)
    eval_env, _ = make_env(args.env_id, seed=args.seed + 1, max_episode_steps=args.max_episode_steps)
    assert_continuous_box_space(train_env)

    agent = build_agent(args, train_env)
    run_name = args.run_name or f"{args.algo}_{safe_name(args.env_id)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{args.seed}"
    run_dir = Path(args.log_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_args(run_dir / "args.json", args)

    metrics_path = run_dir / "metrics.csv"
    fieldnames = [
        "episode",
        "train_return",
        "episode_length",
        "moving_return",
        "eval_return_mean",
        "eval_return_std",
        "policy_loss",
        "value_loss",
        "entropy",
        "total_loss",
        "elapsed_sec",
    ]

    best_eval = -float("inf")
    recent = deque(maxlen=args.moving_average)
    start_time = time.perf_counter()

    with metrics_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()

        progress = trange(
            1,
            args.episodes + 1,
            desc=f"training {args.algo}",
            dynamic_ncols=True,
            disable=args.no_progress or not sys.stderr.isatty(),
        )
        for episode in progress:
            episode_seed = args.seed + episode if args.seed_each_episode else None
            train_return, length = run_training_episode(agent, train_env, seed=episode_seed)
            update_stats = agent.update()
            recent.append(train_return)

            eval_mean = ""
            eval_std = ""
            if args.eval_every > 0 and episode % args.eval_every == 0:
                eval_mean_value, eval_std_value = evaluate(agent, eval_env, args.eval_episodes, seed=args.seed)
                eval_mean = eval_mean_value
                eval_std = eval_std_value
                if eval_mean_value > best_eval:
                    best_eval = eval_mean_value
                    agent.save(run_dir / "best.pt", metadata={"episode": episode, "eval_return_mean": best_eval})

            if args.checkpoint_every > 0 and episode % args.checkpoint_every == 0:
                agent.save(run_dir / f"episode_{episode}.pt", metadata={"episode": episode})

            agent.save(run_dir / "last.pt", metadata={"episode": episode, "backend": backend})

            elapsed = time.perf_counter() - start_time
            row = {
                "episode": episode,
                "train_return": train_return,
                "episode_length": length,
                "moving_return": float(np.mean(recent)),
                "eval_return_mean": eval_mean,
                "eval_return_std": eval_std,
                "policy_loss": update_stats.policy_loss,
                "value_loss": update_stats.value_loss,
                "entropy": update_stats.entropy,
                "total_loss": update_stats.total_loss,
                "elapsed_sec": elapsed,
            }
            writer.writerow(row)
            fp.flush()

            progress.set_postfix(
                {
                    "return": f"{train_return:.1f}",
                    "avg": f"{np.mean(recent):.1f}",
                    "eval": "" if eval_mean == "" else f"{eval_mean:.1f}",
                }
            )

    train_env.close()
    eval_env.close()
    return run_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Train Part 1 policy-gradient agents on Hopper.")
    parser.add_argument("--algo", choices=["reinforce", "reinforce_baseline", "actor_critic"], required=True)
    parser.add_argument("--env-id", default="Hopper-v4")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-each-episode", action="store_true")
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--hidden-sizes", default="64,64")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--baseline", type=float, default=20.0)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--normalize-advantages", action="store_true")
    parser.add_argument("--full-trajectory-return", action="store_true")
    parser.add_argument("--ac-target", choices=["td", "mc"], default="td")
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--moving-average", type=int, default=25)
    parser.add_argument("--log-dir", default="part1/runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    output_dir = train(parse_args())
    print(f"Run saved to: {output_dir}")

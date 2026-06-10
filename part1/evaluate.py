from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from agent import ActorCriticAgent, ReinforceAgent, parse_hidden_sizes, set_seed
from env_utils import assert_continuous_box_space, make_env, reset_env, step_env


def build_agent_from_args(args_payload: dict, env, device: str):
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    hidden_sizes = parse_hidden_sizes(args_payload.get("hidden_sizes", "64,64"))
    common = {
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "action_low": env.action_space.low,
        "action_high": env.action_space.high,
        "hidden_sizes": hidden_sizes,
        "gamma": float(args_payload.get("gamma", 0.99)),
        "entropy_coef": float(args_payload.get("entropy_coef", 0.0)),
        "max_grad_norm": float(args_payload.get("max_grad_norm", 1.0)),
        "normalize_advantages": bool(args_payload.get("normalize_advantages", False)),
        "device": device,
    }
    algo = args_payload["algo"]
    if algo in {"reinforce", "reinforce_baseline"}:
        return ReinforceAgent(
            **common,
            lr=float(args_payload.get("actor_lr", 3e-4)),
            baseline=float(args_payload.get("baseline", 20.0)) if algo == "reinforce_baseline" else 0.0,
            reward_to_go=not bool(args_payload.get("full_trajectory_return", False)),
        )
    if algo == "actor_critic":
        return ActorCriticAgent(
            **common,
            actor_lr=float(args_payload.get("actor_lr", 3e-4)),
            critic_lr=float(args_payload.get("critic_lr", 1e-3)),
            value_coef=float(args_payload.get("value_coef", 0.5)),
            target_mode=args_payload.get("ac_target", "td"),
        )
    raise ValueError(f"Unsupported algorithm: {algo}")


def evaluate(agent, env, episodes: int, seed: int):
    returns: list[float] = []
    lengths: list[int] = []
    for episode in range(episodes):
        obs, _ = reset_env(env, seed=seed + episode)
        done = False
        total = 0.0
        length = 0
        while not done:
            action, _ = agent.select_action(obs, deterministic=True)
            obs, reward, done, _ = step_env(env, action)
            total += reward
            length += 1
        returns.append(total)
        lengths.append(length)
    return returns, lengths


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a Part 1 checkpoint.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    run_dir = Path(args.run_dir)
    args_payload = json.loads((run_dir / "args.json").read_text(encoding="utf-8"))
    env_id = args.env_id or args_payload["env_id"]
    env, _ = make_env(env_id, seed=args.seed, max_episode_steps=args_payload.get("max_episode_steps"))
    assert_continuous_box_space(env)
    agent = build_agent_from_args(args_payload, env, args.device)
    agent.load(run_dir / args.checkpoint)
    returns, lengths = evaluate(agent, env, args.episodes, args.seed)
    env.close()

    print(f"env_id: {env_id}")
    print(f"checkpoint: {run_dir / args.checkpoint}")
    print(f"episodes: {args.episodes}")
    print(f"return_mean: {np.mean(returns):.3f}")
    print(f"return_std: {np.std(returns):.3f}")
    print(f"length_mean: {np.mean(lengths):.3f}")
    print(f"length_std: {np.std(lengths):.3f}")


if __name__ == "__main__":
    main()


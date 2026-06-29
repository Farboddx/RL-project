from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import panda_gym  # noqa: F401 - required so Panda envs are registered
from stable_baselines3 import PPO, SAC


def infer_algo(model_path: str, explicit_algo: str) -> str:
    if explicit_algo != "auto":
        return explicit_algo
    lower_path = model_path.lower()
    if "ppo" in lower_path:
        return "ppo"
    if "sac" in lower_path:
        return "sac"

    metadata_path = Path(model_path).parent.parent / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("algo") in {"ppo", "sac"}:
            return metadata["algo"]

    raise ValueError("Could not infer algorithm. Pass --algo ppo or --algo sac.")


def load_model(model_path: str, algo: str):
    if algo == "ppo":
        return PPO.load(model_path)
    if algo == "sac":
        return SAC.load(model_path)
    raise ValueError("algo must be ppo or sac")


def evaluate(
    model_path: str,
    algo: str,
    n_episodes: int,
    deterministic: bool,
    render: bool,
    env_type: str,
    reward_type: str,
    seed: int,
    output_csv: str | None,
    output_json: str | None,
) -> dict:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Make sure you saved your trained model with model.save(...)."
        )

    resolved_algo = infer_algo(model_path, algo)
    model = load_model(model_path, resolved_algo)
    render_mode = "human" if render else "rgb_array"
    env = gym.make("PandaPush-v3", render_mode=render_mode, type=env_type, reward_type=reward_type)

    episode_rows = []
    returns = []
    successes = []
    lengths = []

    for episode in range(1, n_episodes + 1):
        obs, info = env.reset(seed=seed + episode - 1)
        terminated = False
        truncated = False
        episode_return = 0.0
        length = 0
        final_info = info

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, final_info = env.step(action)
            episode_return += float(reward)
            length += 1

        is_success = float(final_info.get("is_success", 0.0)) if isinstance(final_info, dict) else 0.0
        returns.append(episode_return)
        successes.append(is_success)
        lengths.append(length)
        episode_rows.append(
            {
                "episode": episode,
                "return": episode_return,
                "length": length,
                "is_success": is_success,
            }
        )
        print(f"Episode {episode:03d} | return={episode_return:.3f} | length={length} | success={is_success:.0f}")

    env.close()

    returns_arr = np.array(returns, dtype=np.float32)
    lengths_arr = np.array(lengths, dtype=np.float32)
    successes_arr = np.array(successes, dtype=np.float32)
    summary = {
        "model_path": model_path,
        "algo": resolved_algo,
        "env_type": env_type,
        "reward_type": reward_type,
        "episodes": n_episodes,
        "deterministic": deterministic,
        "seed": seed,
        "mean_return": float(returns_arr.mean()),
        "std_return": float(returns_arr.std()),
        "min_return": float(returns_arr.min()),
        "max_return": float(returns_arr.max()),
        "mean_length": float(lengths_arr.mean()),
        "success_rate": float(successes_arr.mean()),
    }

    print("\n=== Evaluation summary ===")
    print(f"Algorithm: {resolved_algo.upper()}")
    print(f"Environment: {env_type}")
    print(f"Episodes: {n_episodes}")
    print(f"Mean return: {summary['mean_return']:.3f}")
    print(f"Std return:  {summary['std_return']:.3f}")
    print(f"Mean length: {summary['mean_length']:.3f}")
    print(f"Success:     {summary['success_rate']:.2%}")

    if output_csv:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=["episode", "return", "length", "is_success"])
            writer.writeheader()
            writer.writerows(episode_rows)

    if output_json:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PPO/SAC on PandaPush-v3.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--algo", choices=["auto", "ppo", "sac"], default="auto")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--env-type", choices=["source", "target"], default="target")
    parser.add_argument("--reward-type", choices=["dense", "sparse"], default="dense")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        model_path=args.model_path,
        algo=args.algo,
        n_episodes=args.episodes,
        deterministic=not args.stochastic,
        render=args.render,
        env_type=args.env_type,
        reward_type=args.reward_type,
        seed=args.seed,
        output_csv=args.output_csv,
        output_json=args.output_json,
    )


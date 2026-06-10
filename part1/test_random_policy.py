from __future__ import annotations

import argparse

import numpy as np

from env_utils import make_env, reset_env, step_env


def parse_args():
    parser = argparse.ArgumentParser(description="Run a random policy in Hopper.")
    parser.add_argument("--env-id", default="Hopper-v4")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render-mode", default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    env, backend = make_env(
        args.env_id,
        seed=args.seed,
        render_mode=args.render_mode,
        max_episode_steps=args.max_episode_steps,
    )
    returns = []
    lengths = []

    for episode in range(args.episodes):
        obs, _ = reset_env(env, seed=args.seed + episode)
        done = False
        total = 0.0
        length = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, done, _ = step_env(env, action)
            total += reward
            length += 1
        returns.append(total)
        lengths.append(length)
        print(f"episode={episode + 1} return={total:.3f} length={length}")

    env.close()
    print(f"backend: {backend}")
    print(f"mean_return: {np.mean(returns):.3f}")
    print(f"std_return: {np.std(returns):.3f}")
    print(f"mean_length: {np.mean(lengths):.3f}")


if __name__ == "__main__":
    main()


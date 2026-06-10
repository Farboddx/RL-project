from __future__ import annotations

import argparse

import gymnasium as gym
import panda_gym  # noqa: F401 - registers Panda envs


def inspect_env(env_type: str, reward_type: str) -> None:
    env = gym.make("PandaPush-v3", render_mode="rgb_array", type=env_type, reward_type=reward_type)
    task = env.unwrapped.task
    sim = task.sim
    object_id = sim._bodies_idx["object"]
    dynamics = sim.physics_client.getDynamicsInfo(object_id, -1)
    mass = float(dynamics[0])

    print(f"\n[{env_type}]")
    print(f"observation_space: {env.observation_space}")
    print(f"action_space: {env.action_space}")
    print(f"task.current_mass: {task.current_mass:.3f}")
    print(f"pybullet_object_mass: {mass:.3f}")
    print(f"max_episode_steps: 50")
    env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect source and target PandaPush-v3 environments.")
    parser.add_argument("--reward-type", choices=["dense", "sparse"], default="dense")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    inspect_env("source", args.reward_type)
    inspect_env("target", args.reward_type)


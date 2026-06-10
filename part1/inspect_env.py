from __future__ import annotations

import argparse

from env_utils import describe_env, make_env


def print_description(label: str, env_id: str) -> None:
    env, backend = make_env(env_id)
    description = describe_env(env)
    print(f"\n[{label}]")
    print(f"backend: {backend}")
    print(f"env_id: {env_id}")
    print(f"observation_space: {description.observation_space}")
    print(f"action_space: {description.action_space}")
    print(f"nv: {description.nv}")
    print(f"nu: {description.nu}")

    if description.body_names and description.body_masses:
        print("body_masses:")
        for index, mass in enumerate(description.body_masses):
            name = description.body_names[index] if index < len(description.body_names) else f"body_{index}"
            dofs = description.body_dofnum[index] if index < len(description.body_dofnum) else "?"
            print(f"  {index:02d} {name}: mass={mass:.8f}, body_dofnum={dofs}")
    else:
        print("body_masses: unavailable; make sure this is a MuJoCo environment.")

    env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect Hopper source and target environments.")
    parser.add_argument("--source-env-id", default="Hopper-v4")
    parser.add_argument("--target-env-id", default="Hopper-v4")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print_description("source", args.source_env_id)
    print_description("target", args.target_env_id)


from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import panda_gym  # noqa: F401 - required so Panda envs are registered
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from rand_wrapper import RandomizationWrapper


ALGORITHMS = {
    "ppo": PPO,
    "sac": SAC,
}


def parse_range(value: str) -> tuple[float, float]:
    items = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(items) != 2:
        raise argparse.ArgumentTypeError("Range must have exactly two comma-separated floats, e.g. 0.5,8.0")
    return min(items), max(items)


def make_push_env(
    env_type: str,
    reward_type: str,
    sampling_strategy: str,
    mass_range: tuple[float, float],
    adr_initial_range: tuple[float, float],
    adr_step: float,
    adr_boundary_prob: float,
    adr_success_threshold: float,
    seed: int,
    verbose_randomization: bool,
) -> gym.Env:
    env = gym.make(
        "PandaPush-v3",
        render_mode="rgb_array",
        type=env_type,
        reward_type=reward_type,
    )
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    if sampling_strategy != "none":
        env = RandomizationWrapper(
            env,
            mode=sampling_strategy,
            mass_range=mass_range,
            adr_initial_range=adr_initial_range,
            adr_step=adr_step,
            adr_boundary_prob=adr_boundary_prob,
            adr_success_threshold=adr_success_threshold,
            verbose=verbose_randomization,
        )

    return Monitor(env)


def build_model(args: argparse.Namespace, env: gym.Env):
    algo_cls = ALGORITHMS[args.algo]
    common_kwargs = {
        "policy": "MultiInputPolicy",
        "env": env,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "verbose": args.verbose,
        "seed": args.seed,
        "tensorboard_log": str(Path(args.output_dir) / "tensorboard"),
    }

    if args.algo == "ppo":
        return algo_cls(
            **common_kwargs,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
        )

    return algo_cls(
        **common_kwargs,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        tau=args.tau,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO or SAC on PandaPush-v3.")
    parser.add_argument("--algo", choices=["ppo", "sac"], default="sac")
    parser.add_argument("--sampling-strategy", choices=["none", "udr", "adr"], default="none")
    parser.add_argument("--env-type", choices=["source", "target"], default="source")
    parser.add_argument("--reward-type", choices=["dense", "sparse"], default="dense")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--verbose-randomization", action="store_true")

    parser.add_argument("--mass-range", type=parse_range, default=(0.5, 8.0))
    parser.add_argument("--adr-initial-range", type=parse_range, default=(1.0, 2.0))
    parser.add_argument("--adr-step", type=float, default=0.25)
    parser.add_argument("--adr-boundary-prob", type=float, default=0.5)
    parser.add_argument("--adr-success-threshold", type=float, default=0.5)

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=256)

    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)

    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)

    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=50_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)

    run_name = args.run_name or (
        f"{args.algo}_push_{args.sampling_strategy}_{args.env_type}_"
        f"{args.timesteps // 1000}k_seed{args.seed}"
    )
    run_dir = Path(args.output_dir) / run_name
    model_dir = run_dir / "models"
    log_dir = run_dir / "logs"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    train_env = make_push_env(
        env_type=args.env_type,
        reward_type=args.reward_type,
        sampling_strategy=args.sampling_strategy,
        mass_range=args.mass_range,
        adr_initial_range=args.adr_initial_range,
        adr_step=args.adr_step,
        adr_boundary_prob=args.adr_boundary_prob,
        adr_success_threshold=args.adr_success_threshold,
        seed=args.seed,
        verbose_randomization=args.verbose_randomization,
    )
    eval_env = make_push_env(
        env_type="target",
        reward_type=args.reward_type,
        sampling_strategy="none",
        mass_range=args.mass_range,
        adr_initial_range=args.adr_initial_range,
        adr_step=args.adr_step,
        adr_boundary_prob=args.adr_boundary_prob,
        adr_success_threshold=args.adr_success_threshold,
        seed=args.seed + 10_000,
        verbose_randomization=False,
    )

    model = build_model(args, train_env)

    callbacks = []
    if args.checkpoint_every > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=args.checkpoint_every,
                save_path=str(model_dir),
                name_prefix=f"{args.algo}_{args.sampling_strategy}_{args.env_type}",
            )
        )
    if args.eval_every > 0:
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=str(model_dir),
                log_path=str(log_dir),
                eval_freq=args.eval_every,
                n_eval_episodes=args.eval_episodes,
                deterministic=True,
                render=False,
            )
        )

    model.learn(total_timesteps=args.timesteps, callback=callbacks)

    final_model_path = model_dir / "final_model"
    model.save(str(final_model_path))
    best_model_path = model_dir / "best_model"
    if not best_model_path.with_suffix(".zip").exists():
        model.save(str(best_model_path))

    metadata = vars(args).copy()
    metadata.update(
        {
            "run_name": run_name,
            "run_dir": str(run_dir),
            "final_model": str(final_model_path) + ".zip",
            "best_model": str(best_model_path) + ".zip",
            "source_mass": 1.0,
            "target_mass": 5.0,
        }
    )
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    train_env.close()
    eval_env.close()
    print(f"Saved run to: {run_dir}")
    print(f"Final model: {final_model_path}.zip")


if __name__ == "__main__":
    main()

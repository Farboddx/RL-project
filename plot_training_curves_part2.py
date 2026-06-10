from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


RUNS = {
    "ppo_source_none_5000": {
        "tb_dir": "PPO_1",
        "label": "PPO source none",
        "filename": "ppo_source_none_training.png",
    },
    "ppo_target_none_5000": {
        "tb_dir": "PPO_2",
        "label": "PPO target none",
        "filename": "ppo_target_none_training.png",
    },
    "sac_source_none_5000": {
        "tb_dir": "SAC_5",
        "label": "SAC source none",
        "filename": "sac_source_none_training.png",
    },
    "sac_target_none_5000": {
        "tb_dir": "SAC_6",
        "label": "SAC target none",
        "filename": "sac_target_none_training.png",
    },
    "sac_source_udr_5000": {
        "tb_dir": "SAC_7",
        "label": "SAC source UDR",
        "filename": "sac_source_udr_training.png",
    },
    "sac_source_adr_5000": {
        "tb_dir": "SAC_8",
        "label": "SAC source ADR",
        "filename": "sac_source_adr_training.png",
    },
}

SCALARS = {
    "rollout/ep_rew_mean": "Mean Episode Return",
    "rollout/success_rate": "Success Rate",
    "rollout/ep_len_mean": "Mean Episode Length",
    "train/actor_loss": "Actor Loss",
    "train/critic_loss": "Critic Loss",
    "train/ent_coef": "Entropy Coefficient",
}


def load_scalars(tb_dir: Path) -> dict[str, tuple[list[int], list[float]]]:
    accumulator = EventAccumulator(str(tb_dir))
    accumulator.Reload()
    data: dict[str, tuple[list[int], list[float]]] = {}
    for tag in accumulator.Tags().get("scalars", []):
        events = accumulator.Scalars(tag)
        data[tag] = ([event.step for event in events], [event.value for event in events])
    return data


def plot_single_run(name: str, run_info: dict[str, str], data: dict[str, tuple[list[int], list[float]]], out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    fig.suptitle(run_info["label"], fontsize=15, fontweight="bold")

    for ax, (tag, title) in zip(axes.ravel(), SCALARS.items()):
        if tag not in data:
            ax.set_visible(False)
            continue
        steps, values = data[tag]
        ax.plot(steps, values, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Timesteps")
        ax.grid(True, alpha=0.3)
        if tag == "rollout/success_rate":
            ax.set_ylim(0, 1)

    fig.savefig(out_dir / run_info["filename"], dpi=180)
    plt.close(fig)


def plot_comparison(all_data: dict[str, dict[str, tuple[list[int], list[float]]]], out_dir: Path) -> None:
    for tag, title, filename in [
        ("rollout/ep_rew_mean", "Training Mean Episode Return", "part2_training_comparison_reward.png"),
        ("rollout/success_rate", "Training Success Rate", "part2_training_comparison_success.png"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        for key, run_info in RUNS.items():
            data = all_data[key]
            if tag not in data:
                continue
            steps, values = data[tag]
            ax.plot(steps, values, linewidth=2, label=run_info["label"])
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Timesteps")
        ax.grid(True, alpha=0.3)
        if tag == "rollout/success_rate":
            ax.set_ylim(0, 1)
        ax.legend()
        fig.savefig(out_dir / filename, dpi=180)
        plt.close(fig)


def main() -> None:
    tensorboard_dir = Path("runs") / "tensorboard"
    out_dir = Path("runs") / "training_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_data = {}
    for key, run_info in RUNS.items():
        run_dir = tensorboard_dir / run_info["tb_dir"]
        if not run_dir.exists():
            print(f"Skipping missing TensorBoard run: {run_dir}")
            continue
        data = load_scalars(run_dir)
        all_data[key] = data
        plot_single_run(key, run_info, data, out_dir)

    plot_comparison(all_data, out_dir)
    print(f"Saved training plots to: {out_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_input(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        return label, Path(path)
    path = Path(value)
    return path.parent.name, path


def read_metric(path: Path, metric: str):
    episodes = []
    values = []
    with path.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            raw = row.get(metric, "")
            if raw == "":
                continue
            episodes.append(int(row["episode"]))
            values.append(float(raw))
    return np.asarray(episodes), np.asarray(values)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Part 1 training curves.")
    parser.add_argument("--inputs", nargs="+", required=True, help="metrics.csv paths or label=path pairs")
    parser.add_argument("--metric", default="train_return")
    parser.add_argument("--smooth", type=int, default=25)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="Part 1 Hopper training curves")
    return parser.parse_args()


def main():
    args = parse_args()
    plt.figure(figsize=(8, 4.8))
    for item in args.inputs:
        label, path = parse_input(item)
        episodes, values = read_metric(path, args.metric)
        if len(values) == 0:
            continue
        plt.plot(episodes, moving_average(values, args.smooth), label=label)

    plt.title(args.title)
    plt.xlabel("Episode")
    plt.ylabel(args.metric.replace("_", " ").title())
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    print(f"Saved plot to {output}")


if __name__ == "__main__":
    main()

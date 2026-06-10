# FAIML RL Project - Part 2

This folder completes the Part 2 template:

- PPO and SAC training with Stable-Baselines3.
- Source/target baseline evaluation.
- Uniform Domain Randomization.
- Adaptive Domain Randomization.
- Report-ready notes and experiment tracker.

## Setup

From the repository root:

```bash
python3.11 -m venv .venv-part2
source .venv-part2/bin/activate
python -m pip install --upgrade pip
python -m pip install -r part2/requirements_part2.txt
cd part2/panda-gym
python -m pip install -e .
cd ..
```

Use Python 3.10 or 3.11. Python 3.13 is not recommended for Part 2 because `pybullet` may fail to build and the bundled `panda-gym` requires `numpy<2`.

## Inspect Source and Target

From `part2`:

```bash
python inspect_push_env.py
```

Expected cube masses:

- source: `1 kg`
- target: `5 kg`

## Train PPO and SAC

```bash
python train_sb3.py --algo ppo --env-type source --sampling-strategy none --timesteps 500000
python train_sb3.py --algo sac --env-type source --sampling-strategy none --timesteps 500000
```

The model is saved under:

```text
runs/<run-name>/models/
```

## Run the Full Part 2 Matrix

This trains source, target, UDR, and ADR models with SAC by default and evaluates the required configurations, including UDR/ADR on both source and target:

```bash
bash run_part2_matrix.sh 500000 0 sac
```

For quick sanity testing only:

```bash
bash run_part2_matrix.sh 5000 0 sac
```

## Evaluate a Model

```bash
python eval_sb3.py --algo sac --model-path runs/<run-name>/models/best_model.zip --env-type target --episodes 50
```

## Plot Training Curves

After training, generate report-ready PNG plots from TensorBoard logs:

```bash
python plot_training_curves_part2.py
```

## Important Output Files

- `train_sb3.py`: PPO/SAC training.
- `eval_sb3.py`: 50-episode evaluation.
- `rand_wrapper.py`: UDR/ADR mass randomization.
- `inspect_push_env.py`: confirms source/target masses.
- `plot_training_curves_part2.py`: creates training plots for PPO/SAC runs.
- `experiment_tracker_part2.csv`: table to fill with final results.
- `report_part2.md`: text for the report.

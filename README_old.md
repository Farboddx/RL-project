# FAIML RL Project - Part 1

This folder completes Part 1 of the project:

- inspect and test the Gym Hopper environment;
- implement REINFORCE from scratch;
- implement REINFORCE with the required constant baseline `b = 20`;
- implement an Actor-Critic policy-gradient agent from scratch;
- log returns, episode lengths, timing, evaluation scores, and checkpoints;
- provide report-ready notes for the preliminaries and Part 1 analysis.

The code is intentionally independent from Stable-Baselines3 because Part 1 asks for basic RL algorithms implemented from scratch.

## Environment Setup

Use Python 3.10 or 3.11 if possible. Some PyTorch and MuJoCo wheels may not be available for newer Python versions.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r part1/requirements.txt
```

If your machine only has `python3`, replace `python3.11` with `python3`.

## Inspect Hopper

Use this command to answer the environment questions:

```bash
python part1/inspect_env.py --source-env-id Hopper-v4 --target-env-id Hopper-v4
```

If the course repository provides custom source and target Hopper environments, replace the two IDs:

```bash
python part1/inspect_env.py --source-env-id <SOURCE_HOPPER_ID> --target-env-id <TARGET_HOPPER_ID>
```

The script prints:

- observation/state space;
- action space;
- MuJoCo body names;
- body masses;
- number of velocity DoFs;
- number of actuators.

## Random Policy Sanity Check

```bash
python part1/test_random_policy.py --env-id Hopper-v4 --episodes 5
```

This should produce low returns and frequent falls. It is useful only to confirm that the environment can reset, step, and terminate correctly.

## Train REINFORCE

```bash
python part1/train.py \
  --algo reinforce \
  --env-id Hopper-v4 \
  --episodes 1000 \
  --seed 0
```

## Train REINFORCE with Constant Baseline

```bash
python part1/train.py \
  --algo reinforce_baseline \
  --baseline 20 \
  --env-id Hopper-v4 \
  --episodes 1000 \
  --seed 0
```

## Train Actor-Critic

```bash
python part1/train.py \
  --algo actor_critic \
  --env-id Hopper-v4 \
  --episodes 1000 \
  --seed 0
```

For a more sample-efficient Actor-Critic variant, try Monte-Carlo returns as the critic target:

```bash
python part1/train.py \
  --algo actor_critic \
  --ac-target mc \
  --env-id Hopper-v4 \
  --episodes 1000 \
  --seed 0
```

## Run the Full Part 1 Pipeline

After installing dependencies, this command runs environment inspection, random policy testing, all three training jobs, 50-episode evaluation, and the final comparison plot:

```bash
bash part1/run_all_part1.sh Hopper-v4 1000 0
```

## Evaluate a Checkpoint

```bash
python part1/evaluate.py \
  --run-dir part1/runs/<RUN_NAME> \
  --checkpoint best.pt \
  --episodes 50
```

## Plot Training Curves

```bash
python part1/plot_results.py \
  --inputs \
  reinforce=part1/runs/<REINFORCE_RUN>/metrics.csv \
  reinforce_b20=part1/runs/<REINFORCE_BASELINE_RUN>/metrics.csv \
  actor_critic=part1/runs/<ACTOR_CRITIC_RUN>/metrics.csv \
  --output part1/runs/part1_training_curves.png
```

## Expected Report Comparison

Report these metrics for each algorithm:

- final average training return;
- average return over 50 deterministic evaluation episodes;
- episode length;
- wall-clock training time;
- stability of the learning curve.

The usual qualitative expectation is:

- random policy performs poorly;
- REINFORCE without baseline has high variance and unstable learning;
- REINFORCE with `b = 20` can reduce variance when the baseline is close to typical returns, but a fixed baseline is not always optimal;
- Actor-Critic is usually more stable because the critic provides a state-dependent baseline, although it introduces value-function approximation error.
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

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

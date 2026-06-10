#!/usr/bin/env bash
set -euo pipefail

ENV_ID="${1:-Hopper-v4}"
EPISODES="${2:-1000}"
SEED="${3:-0}"

python part1/inspect_env.py --source-env-id "$ENV_ID" --target-env-id "$ENV_ID"
python part1/test_random_policy.py --env-id "$ENV_ID" --episodes 5 --seed "$SEED"

python part1/train.py \
  --algo reinforce \
  --env-id "$ENV_ID" \
  --episodes "$EPISODES" \
  --seed "$SEED" \
  --run-name "reinforce_${ENV_ID}_seed${SEED}"

python part1/train.py \
  --algo reinforce_baseline \
  --baseline 20 \
  --env-id "$ENV_ID" \
  --episodes "$EPISODES" \
  --seed "$SEED" \
  --run-name "reinforce_b20_${ENV_ID}_seed${SEED}"

python part1/train.py \
  --algo actor_critic \
  --env-id "$ENV_ID" \
  --episodes "$EPISODES" \
  --seed "$SEED" \
  --run-name "actor_critic_${ENV_ID}_seed${SEED}"

python part1/evaluate.py \
  --run-dir "part1/runs/reinforce_${ENV_ID}_seed${SEED}" \
  --checkpoint best.pt \
  --episodes 50

python part1/evaluate.py \
  --run-dir "part1/runs/reinforce_b20_${ENV_ID}_seed${SEED}" \
  --checkpoint best.pt \
  --episodes 50

python part1/evaluate.py \
  --run-dir "part1/runs/actor_critic_${ENV_ID}_seed${SEED}" \
  --checkpoint best.pt \
  --episodes 50

python part1/plot_results.py \
  --inputs \
  "reinforce=part1/runs/reinforce_${ENV_ID}_seed${SEED}/metrics.csv" \
  "reinforce_b20=part1/runs/reinforce_b20_${ENV_ID}_seed${SEED}/metrics.csv" \
  "actor_critic=part1/runs/actor_critic_${ENV_ID}_seed${SEED}/metrics.csv" \
  --output "part1/runs/part1_training_curves_${ENV_ID}_seed${SEED}.png"


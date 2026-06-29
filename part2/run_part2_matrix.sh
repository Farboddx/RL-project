#!/usr/bin/env bash
set -euo pipefail

TIMESTEPS="${1:-500000}"
SEED="${2:-0}"
ALG="${3:-sac}"

python inspect_push_env.py

python train_sb3.py \
  --algo "$ALG" \
  --sampling-strategy none \
  --env-type source \
  --timesteps "$TIMESTEPS" \
  --seed "$SEED" \
  --run-name "${ALG}_source_none_${TIMESTEPS}_seed${SEED}"

python train_sb3.py \
  --algo "$ALG" \
  --sampling-strategy none \
  --env-type target \
  --timesteps "$TIMESTEPS" \
  --seed "$SEED" \
  --run-name "${ALG}_target_none_${TIMESTEPS}_seed${SEED}"

python train_sb3.py \
  --algo "$ALG" \
  --sampling-strategy udr \
  --env-type source \
  --mass-range 0.5,8.0 \
  --timesteps "$TIMESTEPS" \
  --seed "$SEED" \
  --run-name "${ALG}_source_udr_${TIMESTEPS}_seed${SEED}"

python train_sb3.py \
  --algo "$ALG" \
  --sampling-strategy adr \
  --env-type source \
  --mass-range 0.5,8.0 \
  --adr-initial-range 1.0,2.0 \
  --adr-step 0.25 \
  --timesteps "$TIMESTEPS" \
  --seed "$SEED" \
  --run-name "${ALG}_source_adr_${TIMESTEPS}_seed${SEED}"

python eval_sb3.py \
  --model-path "runs/${ALG}_source_none_${TIMESTEPS}_seed${SEED}/models/best_model.zip" \
  --algo "$ALG" \
  --env-type source \
  --episodes 50 \
  --output-json "runs/eval_${ALG}_source_to_source.json" \
  --output-csv "runs/eval_${ALG}_source_to_source.csv"

python eval_sb3.py \
  --model-path "runs/${ALG}_source_none_${TIMESTEPS}_seed${SEED}/models/best_model.zip" \
  --algo "$ALG" \
  --env-type target \
  --episodes 50 \
  --output-json "runs/eval_${ALG}_source_to_target.json" \
  --output-csv "runs/eval_${ALG}_source_to_target.csv"

python eval_sb3.py \
  --model-path "runs/${ALG}_target_none_${TIMESTEPS}_seed${SEED}/models/best_model.zip" \
  --algo "$ALG" \
  --env-type target \
  --episodes 50 \
  --output-json "runs/eval_${ALG}_target_to_target.json" \
  --output-csv "runs/eval_${ALG}_target_to_target.csv"

python eval_sb3.py \
  --model-path "runs/${ALG}_source_udr_${TIMESTEPS}_seed${SEED}/models/best_model.zip" \
  --algo "$ALG" \
  --env-type source \
  --episodes 50 \
  --output-json "runs/eval_${ALG}_udr_to_source.json" \
  --output-csv "runs/eval_${ALG}_udr_to_source.csv"

python eval_sb3.py \
  --model-path "runs/${ALG}_source_udr_${TIMESTEPS}_seed${SEED}/models/best_model.zip" \
  --algo "$ALG" \
  --env-type target \
  --episodes 50 \
  --output-json "runs/eval_${ALG}_udr_to_target.json" \
  --output-csv "runs/eval_${ALG}_udr_to_target.csv"

python eval_sb3.py \
  --model-path "runs/${ALG}_source_adr_${TIMESTEPS}_seed${SEED}/models/best_model.zip" \
  --algo "$ALG" \
  --env-type source \
  --episodes 50 \
  --output-json "runs/eval_${ALG}_adr_to_source.json" \
  --output-csv "runs/eval_${ALG}_adr_to_source.csv"

python eval_sb3.py \
  --model-path "runs/${ALG}_source_adr_${TIMESTEPS}_seed${SEED}/models/best_model.zip" \
  --algo "$ALG" \
  --env-type target \
  --episodes 50 \
  --output-json "runs/eval_${ALG}_adr_to_target.json" \
  --output-csv "runs/eval_${ALG}_adr_to_target.csv"

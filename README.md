# Reinforcement Learning for Robotic Control: Policy Gradients and Sim-to-Sim Domain Randomisation

**Group 73 — FAIMDL 2025–2026, Politecnico di Torino**  
s360234 (Nowrouz) · s359032 (Tahmasebi Far) · s359010 (Golkar Khouzani) · s359417 (Chalish Hafshejani)

---

## Overview

This project investigates reinforcement learning for continuous robotic control and sim-to-sim transfer.

- **Part 1** — Three policy-gradient agents (REINFORCE, REINFORCE + baseline, Actor-Critic) implemented from scratch and evaluated on MuJoCo **Hopper-v4**.
- **Part 2** — SAC-based sim-to-sim transfer pipeline on **PandaPush-v3** with Uniform Domain Randomisation (UDR) and Automatic Domain Randomisation (ADR).

---

## Project Structure

```
.
├── part1/
│   ├── agent.py               # REINFORCE, REINFORCE+baseline, Actor-Critic
│   ├── train.py               # Training loop (1000 episodes, seed 0)
│   ├── evaluate.py            # Deterministic evaluation (50 episodes)
│   ├── env_utils.py           # Environment helpers
│   └── runs/                  # Saved checkpoints and metrics
│       ├── reinforce_Hopper-v4_seed0/
│       ├── reinforce_b20_Hopper-v4_seed0/
│       └── actor_critic_Hopper-v4_seed0/
│
├── part2/
│   ├── train_sb3.py           # SAC/PPO training via Stable-Baselines3
│   ├── eval_sb3.py            # Evaluation (50 deterministic episodes → JSON/CSV)
│   ├── rand_wrapper.py        # UDR and ADR gym wrappers
│   ├── run_part2_matrix.sh    # Reproduce all 7 transfer conditions
│   ├── panda_gym_modified/    # Modified panda-gym (adds source/target mass param)
│   │   └── panda_gym/envs/
│   │       ├── panda_tasks.py
│   │       └── tasks/push.py
│   └── runs/                  # Evaluation results and trained models
│       ├── eval_sac_*.json          (7 result files)
│       └── sac_*_1000000_seed0/
│           └── models/best_model.zip
│
└── README.md
```

---

## Part 1 — Policy Gradients on Hopper-v4

### Environment

- **Task:** MuJoCo Hopper-v4 — move a one-legged robot forward without falling
- **Observation:** 11-dimensional continuous Box (positions + velocities)
- **Action:** 3-dimensional continuous Box, clipped to [−1, 1]

### Algorithms

| Method | Description |
|--------|-------------|
| REINFORCE | Monte-Carlo policy gradient with reward-to-go |
| REINFORCE + b=20 | Constant baseline b=20 subtracted to reduce variance |
| Actor-Critic | Stochastic actor + one-step TD value critic |

### Results (seed 0, best checkpoint, 50 deterministic episodes)

| Method | Mean Return | Std Return | Mean Length | Time (s) |
|--------|-------------|------------|-------------|----------|
| Random | 45.6 | 34.1 | 39.0 | 0.0 |
| REINFORCE | **331.5** | 1.6 | 212.5 | 23.2 |
| REINFORCE b=20 | 277.6 | 0.8 | 155.6 | 27.9 |
| Actor-Critic | 228.8 | 1.2 | 100.3 | 32.6 |

### Setup & Run

```bash
cd part1
pip install gymnasium[mujoco] torch numpy

# Train
python train.py --algo reinforce      --seed 0
python train.py --algo reinforce_b    --seed 0
python train.py --algo actor_critic   --seed 0

# Evaluate
python evaluate.py --algo reinforce \
    --checkpoint runs/reinforce_Hopper-v4_seed0/best.pt
```

---

## Part 2 — Sim-to-Sim Transfer on PandaPush-v3

### Transfer Setting

| Domain | Cube Mass |
|--------|-----------|
| Source | 1 kg |
| Target | 5 kg |

A policy trained only in the **source** domain and evaluated on the **target** constitutes the lower bound. Training directly on the target domain is the upper bound.

### Randomisation Strategies

| Strategy | Description |
|----------|-------------|
| **None** | Fixed cube mass during training |
| **UDR** | Mass sampled uniformly from [0.5, 8.0] kg at each episode reset |
| **ADR** | Range starts at [1.0, 2.0] kg; expands/contracts by δ=0.25 kg based on success; global limits [0.5, 8.0] kg |

### Results (SAC, 1 000 000 timesteps, seed 0, 50 deterministic episodes)

| Train → Test | Randomisation | Mean Return | Std Return | Success |
|-------------|---------------|-------------|------------|---------|
| source → source | none | −0.40 | 0.25 | **100%** |
| source → target | none | −0.51 | 0.39 | **100%** |
| target → target | none | −0.38 | 0.23 | **100%** |
| source → source | UDR | −0.39 | 0.25 | **100%** |
| source → target | UDR | −0.43 | 0.27 | **100%** |
| source → source | ADR | −0.41 | 0.25 | **100%** |
| source → target | ADR | −0.45 | 0.28 | **100%** |

SAC achieves **100% success across all seven conditions** at 1 000 000 timesteps. The differences are in mean return (pushing efficiency): UDR largely closes the source→target gap (−0.43 vs −0.40 on source), while ADR produces a slightly more cautious policy (−0.45) due to its gradual curriculum expansion.

### Setup

**Requirements:** Python 3.12

```bash
cd part2

# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install stable-baselines3 gymnasium torch numpy

# Install modified panda-gym (source/target mass parameter)
pip install -e panda_gym_modified/
```

### Run

```bash
# Train and evaluate all 7 conditions in parallel (~2.5 hours on Apple M1)
bash run_part2_matrix.sh

# Train a single condition manually
python train_sb3.py \
    --algo sac \
    --sampling-strategy none \
    --env-type source \
    --timesteps 1000000 \
    --seed 0 \
    --run-name sac_source_none_1000000_seed0

# Evaluate a trained model
python eval_sb3.py \
    --model-path runs/sac_source_none_1000000_seed0/models/best_model.zip \
    --algo sac \
    --env-type target \
    --episodes 50 \
    --output-json runs/eval_sac_source_to_target.json \
    --output-csv  runs/eval_sac_source_to_target.csv
```

### SAC Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 3 × 10⁻⁴ |
| Discount γ | 0.95 |
| Batch size | 256 |
| Replay buffer | 500 000 |
| Learning starts | 1 000 |
| τ (soft update) | 0.005 |

---

## Dependencies

| Package | Version |
|---------|---------|
| Python | 3.12 |
| PyTorch | ≥ 2.0 |
| Stable-Baselines3 | ≥ 2.0 |
| Gymnasium | ≥ 0.29 |
| MuJoCo | ≥ 2.3 |
| panda-gym | bundled (modified) |
| NumPy | ≥ 1.24 |

---

## Course

FAIMDL — Foundations of AI and Machine Deep Learning, 2025–2026  
Politecnico di Torino

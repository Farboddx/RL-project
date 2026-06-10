# Part 1 Final Results

Date run: 2026-06-01

Environment: `Hopper-v4` through Gymnasium 1.3.0 and MuJoCo 3.9.0.

Training setup:

- seed: `0`
- training episodes: `1000`
- hidden layers: `64,64`
- discount factor: `gamma = 0.99`
- evaluation: deterministic policy, `50` episodes, seed `123`
- best checkpoint selected by periodic 5-episode evaluation during training

## Hopper Environment Inspection

Observation/state space:

- `Box(-inf, inf, (11,), float64)`
- Continuous state space.

Action space:

- `Box(-1.0, 1.0, (3,), float32)`
- Continuous action space with 3 actuators.

MuJoCo model:

| Body | Mass | Body DoFs |
|---|---:|---:|
| world | 0.00000000 | 0 |
| torso | 3.66519143 | 3 |
| thigh | 4.05789051 | 1 |
| leg | 2.78135670 | 1 |
| foot | 5.31557477 | 1 |

Other model values:

- `nv = 6`
- `nu = 3`

Note: after checking the official course repository `lambdavi/FAIML-RL-26`, Part 1 uses the standard Gymnasium `Hopper-v4` environment and does not include separate custom Hopper source/target XML files. Therefore, the reported Hopper masses are the official Part 1 masses for the provided template. If an instructor later provides additional custom Hopper IDs, rerun `inspect_env.py` with those IDs.

## Random Policy Baseline

| Episodes | Mean Return | Std Return | Mean Length |
|---:|---:|---:|---:|
| 5 | 45.629 | 34.135 | 39.000 |

## Trained Policy Evaluation

| Algorithm | Training Episodes | Eval Episodes | Mean Return | Std Return | Mean Length | Training Time (s) |
|---|---:|---:|---:|---:|---:|---:|
| REINFORCE | 1000 | 50 | 331.457 | 1.620 | 212.540 | 23.199 |
| REINFORCE + `b=20` | 1000 | 50 | 277.596 | 0.839 | 155.560 | 27.928 |
| Actor-Critic | 1000 | 50 | 228.826 | 1.188 | 100.260 | 32.555 |

## Final Training Snapshot

| Algorithm | Last Episode Return | Last Episode Length | Final 25-Episode Moving Return | Last 5-Episode Eval Mean |
|---|---:|---:|---:|---:|
| REINFORCE | 217.846 | 103 | 225.676 | 240.865 |
| REINFORCE + `b=20` | 192.917 | 86 | 207.835 | 204.486 |
| Actor-Critic | 194.978 | 92 | 189.150 | 220.583 |

## Output Files

- Training curves: `part1/runs/part1_training_curves_Hopper-v4_seed0.png`
- REINFORCE run: `part1/runs/reinforce_Hopper-v4_seed0`
- REINFORCE + baseline run: `part1/runs/reinforce_b20_Hopper-v4_seed0`
- Actor-Critic run: `part1/runs/actor_critic_Hopper-v4_seed0`
- Result tracker: `part1/experiment_tracker.csv`

## Short Analysis

All learned policies substantially outperformed the random policy. In this run, vanilla REINFORCE achieved the highest 50-episode deterministic evaluation return. The constant baseline `b=20` reduced the variance of the evaluation scores but did not improve the final mean return, likely because the fixed baseline was not well matched to the return scale reached later in training. Actor-Critic produced a valid trained policy, but with the default one-step TD critic it underperformed REINFORCE in this short 1000-episode run, suggesting sensitivity to critic accuracy and learning-rate choices.

# Report Notes - Part 2: PPO/SAC Baselines and Domain Randomization

## Environment

The official Part 2 environment is `PandaPush-v3` from the bundled `panda-gym` package. The observation space is a dictionary with `observation`, `achieved_goal`, and `desired_goal`, so the Stable-Baselines3 models use `MultiInputPolicy`.

The sim-to-sim gap is implemented through the cube mass:

| Domain | Cube Mass |
|---|---:|
| Source | 1 kg |
| Target | 5 kg |

This creates the lower-bound transfer setting `source -> target` and the upper-bound setting `target -> target`.

## Task 4: PPO and SAC

The training script `train_sb3.py` supports both PPO and SAC:

```bash
python train_sb3.py --algo ppo --env-type source --sampling-strategy none --timesteps 500000
python train_sb3.py --algo sac --env-type source --sampling-strategy none --timesteps 500000
```

SAC is the recommended default for the final matrix because it is off-policy and more sample-efficient on continuous-control tasks. PPO is included for the mandatory comparison.

For the local 5,000 timestep verification run, PPO and SAC both completed successfully. The short-run evaluation numbers are similar because 5,000 timesteps is only a smoke-test budget for this environment; it is enough to verify the pipeline, but not enough to separate the algorithms reliably. For the final experimental budget, SAC remains the selected algorithm for Tasks 5 and 6.

## Task 5: Lower and Upper Bound Baselines

Train the best algorithm, recommended SAC, on source and target:

```bash
python train_sb3.py --algo sac --env-type source --sampling-strategy none --timesteps 500000
python train_sb3.py --algo sac --env-type target --sampling-strategy none --timesteps 500000
```

Evaluate:

```bash
python eval_sb3.py --algo sac --model-path runs/<SOURCE_RUN>/models/best_model.zip --env-type source --episodes 50
python eval_sb3.py --algo sac --model-path runs/<SOURCE_RUN>/models/best_model.zip --env-type target --episodes 50
python eval_sb3.py --algo sac --model-path runs/<TARGET_RUN>/models/best_model.zip --env-type target --episodes 50
```

Expected interpretation:

- `source -> source`: reference performance in the training simulator.
- `source -> target`: lower bound, because the policy is transferred to a heavier cube without seeing that mass during training.
- `target -> target`: upper bound, because the policy is trained directly on the target dynamics. In real sim-to-real, this is usually impossible or expensive because target interaction means real hardware interaction.

## Task 6: UDR and ADR

UDR samples the cube mass uniformly at each episode:

```bash
python train_sb3.py --algo sac --env-type source --sampling-strategy udr --mass-range 0.5,8.0 --timesteps 500000
```

ADR starts from a narrower range and expands or contracts boundary values depending on success:

```bash
python train_sb3.py --algo sac --env-type source --sampling-strategy adr --mass-range 0.5,8.0 --adr-initial-range 1.0,2.0 --adr-step 0.25 --timesteps 500000
```

Both policies should be evaluated on source and target:

```bash
python eval_sb3.py --algo sac --model-path runs/<UDR_RUN>/models/best_model.zip --env-type source --episodes 50
python eval_sb3.py --algo sac --model-path runs/<UDR_RUN>/models/best_model.zip --env-type target --episodes 50
python eval_sb3.py --algo sac --model-path runs/<ADR_RUN>/models/best_model.zip --env-type source --episodes 50
python eval_sb3.py --algo sac --model-path runs/<ADR_RUN>/models/best_model.zip --env-type target --episodes 50
```

## Result Table Template

| Training -> Test | Algorithm | Randomization | Mean Return | Std Return | Success Rate |
|---|---|---|---:|---:|---:|
| source -> source | SAC | none | TBD | TBD | TBD |
| source -> target | SAC | none | TBD | TBD | TBD |
| target -> target | SAC | none | TBD | TBD | TBD |
| source -> source | SAC | UDR | TBD | TBD | TBD |
| source -> target | SAC | UDR | TBD | TBD | TBD |
| source -> source | SAC | ADR | TBD | TBD | TBD |
| source -> target | SAC | ADR | TBD | TBD | TBD |

## Smoke Run Results

The following sanity-check run was executed locally with SAC for 5,000 timesteps per training condition. This verifies that the implementation, wrappers, training scripts, saved models, and evaluation scripts run end-to-end. These short-run numbers are not the final experimental results; the final matrix should use 500,000 timesteps or the course-specified budget.

| Training -> Test | Algorithm | Randomization | Timesteps | Mean Return | Std Return | Success Rate |
|---|---|---|---:|---:|---:|---:|
| source -> source | PPO | none | 5,000 | -3.618 | 2.114 | 22.00% |
| source -> target | PPO | none | 5,000 | -3.618 | 2.114 | 22.00% |
| target -> target | PPO | none | 5,000 | -3.618 | 2.114 | 22.00% |
| source -> source | SAC | none | 5,000 | -3.618 | 2.114 | 22.00% |
| source -> target | SAC | none | 5,000 | -3.618 | 2.114 | 22.00% |
| target -> target | SAC | none | 5,000 | -3.618 | 2.114 | 22.00% |
| source -> source | SAC | UDR | 5,000 | -3.618 | 2.114 | 22.00% |
| source -> target | SAC | UDR | 5,000 | -3.618 | 2.114 | 22.00% |
| source -> source | SAC | ADR | 5,000 | -3.618 | 2.114 | 22.00% |
| source -> target | SAC | ADR | 5,000 | -3.618 | 2.114 | 22.00% |

Generated files:

```text
runs/eval_ppo_source_to_source.json
runs/eval_ppo_source_to_target.json
runs/eval_ppo_target_to_target.json
runs/eval_sac_source_to_source.json
runs/eval_sac_source_to_target.json
runs/eval_sac_target_to_target.json
runs/eval_sac_udr_to_source.json
runs/eval_sac_udr_to_target.json
runs/eval_sac_adr_to_source.json
runs/eval_sac_adr_to_target.json
```

Training plots:

```text
runs/training_plots/ppo_source_none_training.png
runs/training_plots/ppo_target_none_training.png
runs/training_plots/sac_source_none_training.png
runs/training_plots/sac_target_none_training.png
runs/training_plots/sac_source_udr_training.png
runs/training_plots/sac_source_adr_training.png
runs/training_plots/part2_training_comparison_reward.png
runs/training_plots/part2_training_comparison_success.png
runs/training_plots/sac_training_comparison_reward.png
runs/training_plots/sac_training_comparison_success.png
```

## Discussion Points

- UDR can improve transfer if the target mass lies inside the training distribution and the range is not so wide that learning becomes too hard.
- UDR has an important limitation: the range is manually chosen and can be either too narrow or too broad.
- ADR reduces manual tuning by adapting the range, but it depends on a success criterion and can expand too slowly or too aggressively.
- The source-target gap here is simplified because only cube mass is changed; real sim-to-real transfer includes many more differences such as friction, sensing noise, latency, calibration, and unmodeled contacts.

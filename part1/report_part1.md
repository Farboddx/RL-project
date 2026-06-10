# Report Notes - Part 1

## Preliminaries

The project studies reinforcement learning for robotic control and introduces the sim-to-real transfer problem. In Part 1, the goal is to train a policy for the MuJoCo Hopper environment with basic policy-gradient methods implemented from scratch. The Hopper task is continuous-control: the policy receives a continuous state vector and outputs continuous torques for the actuated joints. The agent is rewarded for moving forward while remaining upright and avoiding excessive control effort.

## Hopper Environment Answers

Run:

```bash
python part1/inspect_env.py --source-env-id <SOURCE_HOPPER_ID> --target-env-id <TARGET_HOPPER_ID>
```

For the standard Gymnasium `Hopper-v4` environment:

- The state/observation space is continuous. It is a `Box` vector with shape `(11,)`, containing body position and velocity information, with the global x-position excluded from the observation.
- The action space is continuous. It is a `Box` vector with shape `(3,)`, corresponding to torque commands for the actuated Hopper joints.
- The number of velocity degrees of freedom and actuators are printed by `inspect_env.py`.
- After checking the official course repository `lambdavi/FAIML-RL-26`, Part 1 uses the standard Gymnasium `Hopper-v4` environment and does not include separate custom Hopper source/target XML files. The measured body masses are: world `0.00000000`, torso `3.66519143`, thigh `4.05789051`, leg `2.78135670`, and foot `5.31557477`.

## REINFORCE

REINFORCE directly optimizes a stochastic policy by estimating the policy gradient from sampled trajectories. For one episode, the reward-to-go is

```text
G_t = sum_{k=t}^{T-1} gamma^(k-t) r_k.
```

The implemented loss is the negative policy-gradient objective:

```text
L_policy(theta) = - sum_t log pi_theta(a_t | s_t) G_t.
```

The implementation uses a Gaussian policy with a neural-network mean and a learned diagonal log standard deviation. Actions are clipped to the environment action bounds before being passed to Hopper.

## REINFORCE with Constant Baseline

The required baseline experiment uses `b = 20`. The implemented objective is:

```text
L_policy(theta) = - sum_t log pi_theta(a_t | s_t) (G_t - b).
```

The baseline does not change the expected policy gradient when it is action-independent, but it can reduce estimator variance. A good constant baseline should be close to the expected return scale. If it is too small, it has little effect; if it is poorly scaled, it may increase variance or slow learning. In practice, a learned state-value baseline is usually preferable, which motivates Actor-Critic methods.

## Actor-Critic

Actor-Critic combines a stochastic actor policy with a learned critic. The actor selects actions, while the critic estimates the value of states and provides a baseline. The default implementation uses a one-step TD target:

```text
y_t = r_t + gamma V_phi(s_{t+1}) (1 - done_t)
A_t = y_t - V_phi(s_t)
```

The actor and critic losses are:

```text
L_actor(theta) = - mean_t log pi_theta(a_t | s_t) stop_gradient(A_t)
L_critic(phi) = mean_t (V_phi(s_t) - y_t)^2
```

The training script also supports Monte-Carlo critic targets with `--ac-target mc`.

## Experimental Protocol

For each method, use the same environment, number of training episodes, random seeds, network size, and evaluation protocol. Recommended settings:

- algorithms: REINFORCE, REINFORCE with `b = 20`, Actor-Critic;
- seeds: at least `0`, ideally `0, 1, 2`;
- evaluation: 50 deterministic episodes using the saved best checkpoint;
- reported metrics: mean return, return standard deviation, episode length, wall-clock training time.

## Result Table Template

| Algorithm | Seed(s) | Eval Episodes | Mean Return | Std Return | Mean Length | Training Time | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Random policy | 0 | 5 | 45.629 | 34.135 | 39.000 | 0.000s | Environment sanity check |
| REINFORCE | 0 | 50 | 331.457 | 1.620 | 212.540 | 23.199s | Best result in this run |
| REINFORCE + b=20 | 0 | 50 | 277.596 | 0.839 | 155.560 | 27.928s | Constant baseline |
| Actor-Critic | 0 | 50 | 228.826 | 1.188 | 100.260 | 32.555s | State-dependent baseline |

## Discussion Points

- REINFORCE without a baseline usually has high variance because the full return estimate depends strongly on sampled trajectories.
- The constant baseline can reduce variance only if it is near the return scale. It is simple but not adaptive.
- Actor-Critic usually learns more smoothly because the critic gives a state-dependent estimate of expected return.
- Actor-Critic can still become unstable if the critic is inaccurate, the learning rates are too large, or the policy variance collapses too early.
- Hopper is a challenging continuous-control task, so learning curves should be interpreted across multiple seeds rather than from a single run.

In the completed seed-0 run, all learned policies clearly outperform the random baseline. Vanilla REINFORCE obtained the highest deterministic 50-episode evaluation return. The fixed baseline reduced the reported evaluation variance but did not improve mean return, which is plausible because a constant baseline cannot track the changing return scale during training. Actor-Critic learned a working policy but underperformed the REINFORCE variants with the default one-step TD critic, indicating sensitivity to critic accuracy and learning-rate settings.

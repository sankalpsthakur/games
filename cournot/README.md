# Cournot

## Game Spec
- State: demand parameters, marginal cost estimate, and prior quantity profile.
- Action: each firm chooses production quantity.
- Reward: quantity times market price minus production cost.
- Mock video tells: order-entry speed, confidence score on high output.

## Commands
- Train: `python -m cournot.train`
- Eval: `python -m cournot.train --train-episodes 0 --eval-episodes 1000 --video false`
- Visualize: `python -m cournot.train --seeds 1 --train-episodes 50 --eval-episodes 20 --video true`

## Outputs
- Default output location: `cournot/outputs/`
- Override with: `--output-dir /absolute/path`

## Results (Multi-Method Framework)

30 seeds, 280 train steps, 140 eval steps.

| Algorithm | Mean | SD | SE | 95% CI | p vs zero |
|---|---:|---:|---:|---|---:|
| PPO | -0.3737 | 0.2842 | 0.0519 | [-0.4754, -0.2720] | 5.92e-13 |
| **A2C** | **-0.3401** | 0.2868 | 0.0524 | [-0.4427, -0.2375] | 8.30e-11 |
| DQN | -0.3741 | 0.3048 | 0.0557 | [-0.4832, -0.2650] | 1.80e-11 |

Best algorithm: **A2C** (mean -0.3401). All algorithms significantly negative. Pairwise differences not significant.

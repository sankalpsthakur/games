# Ultimatum

## Game Spec
- State: proposer budget, round index, fairness belief about responder.
- Action: proposer chooses split offer; responder accepts or rejects.
- Reward: accepted offers pay both sides by split; rejected offers pay zero.
- Mock video tells: pause-before-accept, eyebrow raise at low offers.

## Commands
- Train: `python -m ultimatum.train`
- Eval: `python -m ultimatum.train --train-episodes 0 --eval-episodes 1000 --video false`
- Visualize: `python -m ultimatum.train --seeds 1 --train-episodes 50 --eval-episodes 20 --video true`

## Outputs
- Default output location: `ultimatum/outputs/`
- Override with: `--output-dir /absolute/path`

## Results (Multi-Method Framework)

30 seeds, 280 train steps, 140 eval steps.

| Algorithm | Mean | SD | SE | 95% CI | p vs zero |
|---|---:|---:|---:|---|---:|
| PPO | -0.3881 | 0.3017 | 0.0551 | [-0.4961, -0.2801] | 1.85e-12 |
| **A2C** | **-0.3515** | 0.3334 | 0.0609 | [-0.4708, -0.2321] | 7.79e-09 |
| DQN | -0.3642 | 0.3075 | 0.0561 | [-0.4742, -0.2541] | 8.78e-11 |

Best algorithm: **A2C** (mean -0.3515). All algorithms significantly negative. Pairwise differences not significant.

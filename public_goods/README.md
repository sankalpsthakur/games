# Public Goods

## Game Spec
- State: player endowment, shared pot, and contribution history.
- Action: contribute a discrete amount to the public pool.
- Reward: kept private amount plus redistributed public return.
- Mock video tells: side-glance frequency, hand-raise confidence before contribution.

## Commands
- Train: `python -m public_goods.train`
- Eval: `python -m public_goods.train --train-episodes 0 --eval-episodes 1000 --video false`
- Visualize: `python -m public_goods.train --seeds 1 --train-episodes 50 --eval-episodes 20 --video true`

## Outputs
- Default output location: `public_goods/outputs/`
- Override with: `--output-dir /absolute/path`

## Results (Multi-Method Framework)

30 seeds, 280 train steps, 140 eval steps.

| Algorithm | Mean | SD | SE | 95% CI | p vs zero |
|---|---:|---:|---:|---|---:|
| **PPO** | **-0.3819** | 0.3500 | 0.0639 | [-0.5071, -0.2566] | 2.29e-09 |
| A2C | -0.4075 | 0.3346 | 0.0611 | [-0.5272, -0.2877] | 2.56e-11 |
| DQN | -0.4007 | 0.2485 | 0.0454 | [-0.4896, -0.3118] | 0.00e+00 |

Best algorithm: **PPO** (mean -0.3819). All algorithms produce strongly negative reward. Pairwise differences not significant.

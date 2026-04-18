# Beer Distribution

## Game Spec
- State: inventory by stage, backlog, and shipment delays.
- Action: each stage submits reorder quantity.
- Reward: negative holding and backlog costs over the horizon.
- Mock video tells: rushed order clicks, fatigue head tilt after stockouts.

## Commands
- Train: `python -m beer_distribution.train`
- Eval: `python -m beer_distribution.train --train-episodes 0 --eval-episodes 1000 --video false`
- Visualize: `python -m beer_distribution.train --seeds 1 --train-episodes 50 --eval-episodes 20 --video true`

## Outputs
- Default output location: `beer_distribution/outputs/`
- Override with: `--output-dir /absolute/path`

## Results (Multi-Method Framework)

30 seeds, 280 train steps, 140 eval steps.

| Algorithm | Mean | SD | SE | 95% CI | p vs zero |
|---|---:|---:|---:|---|---:|
| PPO | -0.2436 | 0.3810 | 0.0696 | [-0.3799, -0.1072] | 4.62e-04 |
| A2C | -0.2572 | 0.3757 | 0.0686 | [-0.3917, -0.1228] | 1.77e-04 |
| **DQN** | **-0.2242** | 0.3646 | 0.0666 | [-0.3547, -0.0937] | 7.58e-04 |

Best algorithm: **DQN** (mean -0.2242). All algorithms negative; pairwise differences not significant.

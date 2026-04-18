# Auctions

## Game Spec
- State: private valuation, auction format metadata, and bid history.
- Action: submit bid or stay out.
- Reward: winner earns value minus payment; losers earn zero.
- Mock video tells: paddle raise speed, voice pitch shift near reserve.

## Commands
- Train: `python -m auctions.train`
- Eval: `python -m auctions.train --train-episodes 0 --eval-episodes 1000 --video false`
- Visualize: `python -m auctions.train --seeds 1 --train-episodes 50 --eval-episodes 20 --video true`

## Outputs
- Default output location: `auctions/outputs/`
- Override with: `--output-dir /absolute/path`

## Results (Multi-Method Framework)

30 seeds, 280 train steps, 140 eval steps.

| Algorithm | Mean | SD | SE | 95% CI | p vs zero |
|---|---:|---:|---:|---|---:|
| **PPO** | **-0.3370** | 0.2978 | 0.0544 | [-0.4436, -0.2305] | 5.70e-10 |
| A2C | -0.3620 | 0.2516 | 0.0459 | [-0.4521, -0.2720] | 3.11e-15 |
| DQN | -0.3673 | 0.2851 | 0.0521 | [-0.4693, -0.2652] | 1.72e-12 |

Best algorithm: **PPO** (mean -0.3370). All three algorithms produce significantly negative reward. Pairwise differences are small and mostly not significant.

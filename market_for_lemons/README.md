# Market For Lemons

## Game Spec
- State: hidden seller quality type, buyer prior quality belief, and warranty options.
- Action: seller sets ask/warranty; buyer buys or declines.
- Reward: seller earns price minus quality cost; buyer earns realized quality value minus price.
- Mock video tells: hesitation before warranty, confidence tone on quality claims.

## Commands
- Train: `python -m market_for_lemons.train`
- Eval: `python -m market_for_lemons.train --train-episodes 0 --eval-episodes 1000 --video false`
- Visualize: `python -m market_for_lemons.train --seeds 1 --train-episodes 50 --eval-episodes 20 --video true`

## Outputs
- Default output location: `market_for_lemons/outputs/`
- Override with: `--output-dir /absolute/path`

## Results (Multi-Method Framework)

30 seeds, 280 train steps, 140 eval steps.

| Algorithm | Mean | SD | SE | 95% CI | p vs zero |
|---|---:|---:|---:|---|---:|
| PPO | -0.2931 | 0.3540 | 0.0646 | [-0.4198, -0.1664] | 5.75e-06 |
| A2C | -0.2886 | 0.3231 | 0.0590 | [-0.4042, -0.1730] | 1.00e-06 |
| **DQN** | **-0.2581** | 0.3395 | 0.0620 | [-0.3796, -0.1366] | 3.13e-05 |

Best algorithm: **DQN** (mean -0.2581). All algorithms negative. Pairwise differences not significant.

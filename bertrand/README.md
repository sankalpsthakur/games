# Bertrand

## Game Spec
- State: cost floor, competitor price belief, and demand sensitivity estimate.
- Action: each firm posts a price.
- Reward: margin on captured demand share.
- Mock video tells: price-change hesitation, eye-contact hold before undercutting.

## Commands
- Train: `python -m bertrand.train`
- Eval: `python -m bertrand.train --train-episodes 0 --eval-episodes 1000 --video false`
- Visualize: `python -m bertrand.train --seeds 1 --train-episodes 50 --eval-episodes 20 --video true`

## Outputs
- Default output location: `bertrand/outputs/`
- Override with: `--output-dir /absolute/path`

## Results (Multi-Method Framework)

30 seeds, 280 train steps, 140 eval steps.

| Algorithm | Mean | SD | SE | 95% CI | p vs zero |
|---|---:|---:|---:|---|---:|
| PPO | -0.3820 | 0.2942 | 0.0537 | [-0.4873, -0.2767] | 1.14e-12 |
| A2C | -0.3858 | 0.2827 | 0.0516 | [-0.4869, -0.2847] | 7.66e-14 |
| **DQN** | **-0.3741** | 0.2868 | 0.0524 | [-0.4767, -0.2714] | 9.06e-13 |

Best algorithm: **DQN** (mean -0.3741). All algorithms produce strongly negative reward with tight confidence intervals. Pairwise differences are negligible.

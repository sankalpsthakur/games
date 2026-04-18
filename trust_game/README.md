# Trust Game

## Game Spec
- State: investor endowment, trustee reciprocity belief, and prior repayment ratio.
- Action: investor sends amount; trustee decides return fraction.
- Reward: investor earns leftover plus return; trustee earns received amount minus return.
- Mock video tells: smile duration, return delay before generous repayments.

## Commands
- Train: `python -m trust_game.train`
- Eval: `python -m trust_game.train --train-episodes 0 --eval-episodes 1000 --video false`
- Visualize: `python -m trust_game.train --seeds 1 --train-episodes 50 --eval-episodes 20 --video true`

## Outputs
- Default output location: `trust_game/outputs/`
- Override with: `--output-dir /absolute/path`

## Results (Multi-Method Framework)

30 seeds, 280 train steps, 140 eval steps.

| Algorithm | Mean | SD | SE | 95% CI | p vs zero |
|---|---:|---:|---:|---|---:|
| PPO | -0.3607 | 0.3235 | 0.0591 | [-0.4765, -0.2450] | 1.01e-09 |
| **A2C** | **-0.3141** | 0.3327 | 0.0607 | [-0.4332, -0.1951] | 2.33e-07 |
| DQN | -0.3291 | 0.3396 | 0.0620 | [-0.4506, -0.2076] | 1.11e-07 |

Best algorithm: **A2C** (mean -0.3141). All negative. PPO vs A2C pairwise difference significant (p=0.026).

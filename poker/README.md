# Poker

## Game Spec
- State: hand cards, community cards, pot size, opponent betting history.
- Action: fold, check/call, raise (discrete levels).
- Reward: net chip gain/loss at showdown.
- Mock video tells: bet timing hesitation, chip-handling patterns, gaze direction.

## Implementations
- `poker_experiment.py` -- original experiment script
- `realistic/` -- realistic Hold'em environment with opponent intel and significance evaluation

## Results (Multi-Method Framework)

30 seeds, 280 train steps, 140 eval steps.

| Algorithm | Mean | SD | SE | 95% CI | p vs zero |
|---|---:|---:|---:|---|---:|
| PPO | -0.3071 | 0.3545 | 0.0647 | [-0.4340, -0.1803] | 2.08e-06 |
| **A2C** | **-0.2820** | 0.3519 | 0.0642 | [-0.4079, -0.1561] | 1.13e-05 |
| DQN | -0.2851 | 0.3569 | 0.0652 | [-0.4128, -0.1574] | 1.21e-05 |

Best algorithm: **A2C** (mean -0.2820). All algorithms produce significantly negative reward. Pairwise differences not significant.

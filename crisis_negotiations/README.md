# Crisis Negotiations

## Game Spec
- State: stakes, time remaining, trust estimate, and concession history.
- Action: each side chooses escalation or concession level.
- Reward: settlement utility minus escalation cost; breakdown incurs large penalties.
- Mock video tells: speech rate jump, hand tremor index during brinkmanship.

## Commands
- Train: `python -m crisis_negotiations.train`
- Eval: `python -m crisis_negotiations.train --train-episodes 0 --eval-episodes 1000 --video false`
- Visualize: `python -m crisis_negotiations.train --seeds 1 --train-episodes 50 --eval-episodes 20 --video true`

## Outputs
- Default output location: `crisis_negotiations/outputs/`
- Override with: `--output-dir /absolute/path`

## Results (Multi-Method Framework)

30 seeds, 280 train steps, 140 eval steps.

| Algorithm | Mean | SD | SE | 95% CI | p vs zero |
|---|---:|---:|---:|---|---:|
| PPO | -0.2035 | 0.3778 | 0.0690 | [-0.3387, -0.0683] | 3.18e-03 |
| **A2C** | **-0.1977** | 0.3889 | 0.0710 | [-0.3368, -0.0585] | 5.37e-03 |
| DQN | -0.2138 | 0.3253 | 0.0594 | [-0.3302, -0.0974] | 3.19e-04 |

Best algorithm: **A2C** (mean -0.1977). Negative but closer to zero than most games. Pairwise differences not significant.

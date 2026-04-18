# Stag Hunt

## Game Spec
- State: partner reliability belief, coordination history, and current round.
- Action: choose Stag (risky cooperative) or Hare (safe solo).
- Reward: both get high payoff on Stag/Stag; mismatch rewards Hare hunter only.
- Mock video tells: confident nod before Stag, hesitation frames before Hare.

## Commands
- Train: `python -m stag_hunt.train`
- Eval: `python -m stag_hunt.train --train-episodes 0 --eval-episodes 1000 --video false`
- Visualize: `python -m stag_hunt.train --seeds 1 --train-episodes 50 --eval-episodes 20 --video true`

## Outputs
- Default output location: `stag_hunt/outputs/`
- Override with: `--output-dir /absolute/path`

## Results (Multi-Method Framework)

30 seeds, 280 train steps, 140 eval steps.

| Algorithm | Mean | SD | SE | 95% CI | p vs zero |
|---|---:|---:|---:|---|---:|
| **PPO** | **+0.2959** | 0.5049 | 0.0922 | [+0.1152, +0.4765] | 1.33e-03 |
| A2C | +0.2790 | 0.5046 | 0.0921 | [+0.0984, +0.4595] | 2.46e-03 |
| DQN | +0.2927 | 0.3900 | 0.0712 | [+0.1532, +0.4323] | 3.94e-05 |

Best algorithm: **PPO** (mean +0.2959). One of 3 games with positive reward. All algorithms achieve significant positive means. DQN has the narrowest CI.

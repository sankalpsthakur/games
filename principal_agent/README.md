# Principal Agent

## Game Spec
- State: contract terms, effort proxy, and shirking prior.
- Action: principal sets wage/bonus; agent chooses effort level.
- Reward: principal gets output minus compensation; agent gets compensation minus effort cost.
- Mock video tells: compliance glance, fatigue marker after high effort.

## Commands
- Train: `python -m principal_agent.train`
- Eval: `python -m principal_agent.train --train-episodes 0 --eval-episodes 1000 --video false`
- Visualize: `python -m principal_agent.train --seeds 1 --train-episodes 50 --eval-episodes 20 --video true`

## Outputs
- Default output location: `principal_agent/outputs/`
- Override with: `--output-dir /absolute/path`

## Results (Multi-Method Framework)

30 seeds, 280 train steps, 140 eval steps.

| Algorithm | Mean | SD | SE | 95% CI | p vs zero |
|---|---:|---:|---:|---|---:|
| PPO | -0.2489 | 0.3045 | 0.0556 | [-0.3579, -0.1399] | 7.58e-06 |
| A2C | -0.2506 | 0.3419 | 0.0624 | [-0.3730, -0.1283] | 5.94e-05 |
| **DQN** | **-0.2486** | 0.3132 | 0.0572 | [-0.3607, -0.1365] | 1.38e-05 |

Best algorithm: **DQN** (mean -0.2486). All algorithms very close; differences are negligible (largest pairwise p=0.93).

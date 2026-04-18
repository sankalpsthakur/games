# Interrogation

## Game Spec
- State: evidence strength, hidden suspect type, and answer history.
- Action: interrogator picks pressure/questions; suspect confesses, denies, or partially admits.
- Reward: interrogator gains from true resolution with false-confession penalties; suspect gains from favorable outcome.
- Mock video tells: micro-expression spike, eye aversion during high pressure.

## Commands
- Train: `python -m interrogation.train`
- Eval: `python -m interrogation.train --train-episodes 0 --eval-episodes 1000 --video false`
- Visualize: `python -m interrogation.train --seeds 1 --train-episodes 50 --eval-episodes 20 --video true`

## Outputs
- Default output location: `interrogation/outputs/`
- Override with: `--output-dir /absolute/path`

## Results (Multi-Method Framework)

30 seeds, 280 train steps, 140 eval steps.

| Algorithm | Mean | SD | SE | 95% CI | p vs zero |
|---|---:|---:|---:|---|---:|
| PPO | -0.3882 | 0.2984 | 0.0545 | [-0.4950, -0.2814] | 1.04e-12 |
| A2C | -0.3540 | 0.3298 | 0.0602 | [-0.4720, -0.2360] | 4.14e-09 |
| **DQN** | **-0.3478** | 0.2931 | 0.0535 | [-0.4527, -0.2429] | 8.08e-11 |

Best algorithm: **DQN** (mean -0.3478). All algorithms produce strongly negative reward. Pairwise differences not significant.

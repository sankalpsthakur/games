# Iterated Prisoners Dilemma

## Game Spec
- State: last actions, retaliation streak, and round index.
- Action: cooperate or defect each round.
- Reward: standard dilemma matrix across repeated interactions.
- Mock video tells: blink-rate shift before defection, response delay before cooperation.

## Commands
- Train: `python -m iterated_prisoners_dilemma.train`
- Eval: `python -m iterated_prisoners_dilemma.train --train-episodes 0 --eval-episodes 1000 --video false`
- Visualize: `python -m iterated_prisoners_dilemma.train --seeds 1 --train-episodes 50 --eval-episodes 20 --video true`

## Outputs
- Default output location: `iterated_prisoners_dilemma/outputs/`
- Override with: `--output-dir /absolute/path`

## Results (Multi-Method Framework)

30 seeds, 280 train steps, 140 eval steps.

| Algorithm | Mean | SD | SE | 95% CI | p vs zero |
|---|---:|---:|---:|---|---:|
| **PPO** | **+0.4540** | 0.5140 | 0.0939 | [+0.2700, +0.6379] | 1.32e-06 |
| A2C | +0.4522 | 0.5095 | 0.0930 | [+0.2699, +0.6345] | 1.16e-06 |
| DQN | +0.4471 | 0.4926 | 0.0899 | [+0.2708, +0.6233] | 6.67e-07 |

Best algorithm: **PPO** (mean +0.4540). Highest positive reward across all 18 games. All algorithms perform well; differences are negligible.

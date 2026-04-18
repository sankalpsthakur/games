# Hawk Dove

## Game Spec
- State: resource value, conflict cost, and opponent aggression belief.
- Action: each player chooses Hawk or Dove.
- Reward: Hawk/Hawk splits value with injury cost; Hawk/Dove gives Hawk the pot; Dove/Dove splits peacefully.
- Mock video tells: posture stiffness, gaze hold before choosing Hawk.

## Commands
- Train: `python -m hawk_dove.train`
- Eval: `python -m hawk_dove.train --train-episodes 0 --eval-episodes 1000 --video false`
- Visualize: `python -m hawk_dove.train --seeds 1 --train-episodes 50 --eval-episodes 20 --video true`

## Outputs
- Default output location: `hawk_dove/outputs/`
- Override with: `--output-dir /absolute/path`

## Results (Multi-Method Framework)

30 seeds, 280 train steps, 140 eval steps.

| Algorithm | Mean | SD | SE | 95% CI | p vs zero |
|---|---:|---:|---:|---|---:|
| PPO | +0.2933 | 0.5122 | 0.0935 | [+0.1100, +0.4766] | 1.71e-03 |
| A2C | +0.2951 | 0.5068 | 0.0925 | [+0.1137, +0.4765] | 1.43e-03 |
| **DQN** | **+0.3156** | 0.4178 | 0.0763 | [+0.1661, +0.4651] | 3.51e-05 |

Best algorithm: **DQN** (mean +0.3156). One of only 3 games with positive reward. All algorithms achieve statistically significant positive means. DQN has the tightest CI.

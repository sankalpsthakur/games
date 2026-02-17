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

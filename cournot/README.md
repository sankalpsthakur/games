# Cournot

## Game Spec
- State: demand parameters, marginal cost estimate, and prior quantity profile.
- Action: each firm chooses production quantity.
- Reward: quantity times market price minus production cost.
- Mock video tells: order-entry speed, confidence score on high output.

## Commands
- Train: `python -m cournot.train`
- Eval: `python -m cournot.train --train-episodes 0 --eval-episodes 1000 --video false`
- Visualize: `python -m cournot.train --seeds 1 --train-episodes 50 --eval-episodes 20 --video true`

## Outputs
- Default output location: `cournot/outputs/`
- Override with: `--output-dir /absolute/path`

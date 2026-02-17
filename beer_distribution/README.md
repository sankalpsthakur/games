# Beer Distribution

## Game Spec
- State: inventory by stage, backlog, and shipment delays.
- Action: each stage submits reorder quantity.
- Reward: negative holding and backlog costs over the horizon.
- Mock video tells: rushed order clicks, fatigue head tilt after stockouts.

## Commands
- Train: `python -m beer_distribution.train`
- Eval: `python -m beer_distribution.train --train-episodes 0 --eval-episodes 1000 --video false`
- Visualize: `python -m beer_distribution.train --seeds 1 --train-episodes 50 --eval-episodes 20 --video true`

## Outputs
- Default output location: `beer_distribution/outputs/`
- Override with: `--output-dir /absolute/path`

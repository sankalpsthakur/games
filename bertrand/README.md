# Bertrand

## Game Spec
- State: cost floor, competitor price belief, and demand sensitivity estimate.
- Action: each firm posts a price.
- Reward: margin on captured demand share.
- Mock video tells: price-change hesitation, eye-contact hold before undercutting.

## Commands
- Train: `python -m bertrand.train`
- Eval: `python -m bertrand.train --train-episodes 0 --eval-episodes 1000 --video false`
- Visualize: `python -m bertrand.train --seeds 1 --train-episodes 50 --eval-episodes 20 --video true`

## Outputs
- Default output location: `bertrand/outputs/`
- Override with: `--output-dir /absolute/path`

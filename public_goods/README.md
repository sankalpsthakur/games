# Public Goods

## Game Spec
- State: player endowment, shared pot, and contribution history.
- Action: contribute a discrete amount to the public pool.
- Reward: kept private amount plus redistributed public return.
- Mock video tells: side-glance frequency, hand-raise confidence before contribution.

## Commands
- Train: `python -m public_goods.train`
- Eval: `python -m public_goods.train --train-episodes 0 --eval-episodes 1000 --video false`
- Visualize: `python -m public_goods.train --seeds 1 --train-episodes 50 --eval-episodes 20 --video true`

## Outputs
- Default output location: `public_goods/outputs/`
- Override with: `--output-dir /absolute/path`

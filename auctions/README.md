# Auctions

## Game Spec
- State: private valuation, auction format metadata, and bid history.
- Action: submit bid or stay out.
- Reward: winner earns value minus payment; losers earn zero.
- Mock video tells: paddle raise speed, voice pitch shift near reserve.

## Commands
- Train: `python -m auctions.train`
- Eval: `python -m auctions.train --train-episodes 0 --eval-episodes 1000 --video false`
- Visualize: `python -m auctions.train --seeds 1 --train-episodes 50 --eval-episodes 20 --video true`

## Outputs
- Default output location: `auctions/outputs/`
- Override with: `--output-dir /absolute/path`

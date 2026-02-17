# Market For Lemons

## Game Spec
- State: hidden seller quality type, buyer prior quality belief, and warranty options.
- Action: seller sets ask/warranty; buyer buys or declines.
- Reward: seller earns price minus quality cost; buyer earns realized quality value minus price.
- Mock video tells: hesitation before warranty, confidence tone on quality claims.

## Commands
- Train: `python -m market_for_lemons.train`
- Eval: `python -m market_for_lemons.train --train-episodes 0 --eval-episodes 1000 --video false`
- Visualize: `python -m market_for_lemons.train --seeds 1 --train-episodes 50 --eval-episodes 20 --video true`

## Outputs
- Default output location: `market_for_lemons/outputs/`
- Override with: `--output-dir /absolute/path`

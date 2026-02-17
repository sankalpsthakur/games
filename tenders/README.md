# Tenders

## Game Spec
- State: project value, private cost estimate, and competitor win-rate belief.
- Action: submit bid amount or skip tender.
- Reward: winner gets project margin; losers get zero.
- Mock video tells: bid-submit delay, confidence smile before aggressive bids.

## Commands
- Train: `python -m tenders.train`
- Eval: `python -m tenders.train --train-episodes 0 --eval-episodes 1000 --video false`
- Visualize: `python -m tenders.train --seeds 1 --train-episodes 50 --eval-episodes 20 --video true`

## Outputs
- Default output location: `tenders/outputs/`
- Override with: `--output-dir /absolute/path`

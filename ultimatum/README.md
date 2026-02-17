# Ultimatum

## Game Spec
- State: proposer budget, round index, fairness belief about responder.
- Action: proposer chooses split offer; responder accepts or rejects.
- Reward: accepted offers pay both sides by split; rejected offers pay zero.
- Mock video tells: pause-before-accept, eyebrow raise at low offers.

## Commands
- Train: `python -m ultimatum.train`
- Eval: `python -m ultimatum.train --train-episodes 0 --eval-episodes 1000 --video false`
- Visualize: `python -m ultimatum.train --seeds 1 --train-episodes 50 --eval-episodes 20 --video true`

## Outputs
- Default output location: `ultimatum/outputs/`
- Override with: `--output-dir /absolute/path`

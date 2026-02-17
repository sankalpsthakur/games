# Crisis Negotiations

## Game Spec
- State: stakes, time remaining, trust estimate, and concession history.
- Action: each side chooses escalation or concession level.
- Reward: settlement utility minus escalation cost; breakdown incurs large penalties.
- Mock video tells: speech rate jump, hand tremor index during brinkmanship.

## Commands
- Train: `python -m crisis_negotiations.train`
- Eval: `python -m crisis_negotiations.train --train-episodes 0 --eval-episodes 1000 --video false`
- Visualize: `python -m crisis_negotiations.train --seeds 1 --train-episodes 50 --eval-episodes 20 --video true`

## Outputs
- Default output location: `crisis_negotiations/outputs/`
- Override with: `--output-dir /absolute/path`

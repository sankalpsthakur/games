# Iterated Prisoners Dilemma

## Game Spec
- State: last actions, retaliation streak, and round index.
- Action: cooperate or defect each round.
- Reward: standard dilemma matrix across repeated interactions.
- Mock video tells: blink-rate shift before defection, response delay before cooperation.

## Commands
- Train: `python -m iterated_prisoners_dilemma.train`
- Eval: `python -m iterated_prisoners_dilemma.train --train-episodes 0 --eval-episodes 1000 --video false`
- Visualize: `python -m iterated_prisoners_dilemma.train --seeds 1 --train-episodes 50 --eval-episodes 20 --video true`

## Outputs
- Default output location: `iterated_prisoners_dilemma/outputs/`
- Override with: `--output-dir /absolute/path`

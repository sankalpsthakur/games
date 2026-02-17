# Interrogation

## Game Spec
- State: evidence strength, hidden suspect type, and answer history.
- Action: interrogator picks pressure/questions; suspect confesses, denies, or partially admits.
- Reward: interrogator gains from true resolution with false-confession penalties; suspect gains from favorable outcome.
- Mock video tells: micro-expression spike, eye aversion during high pressure.

## Commands
- Train: `python -m interrogation.train`
- Eval: `python -m interrogation.train --train-episodes 0 --eval-episodes 1000 --video false`
- Visualize: `python -m interrogation.train --seeds 1 --train-episodes 50 --eval-episodes 20 --video true`

## Outputs
- Default output location: `interrogation/outputs/`
- Override with: `--output-dir /absolute/path`

# Trust Game

## Game Spec
- State: investor endowment, trustee reciprocity belief, and prior repayment ratio.
- Action: investor sends amount; trustee decides return fraction.
- Reward: investor earns leftover plus return; trustee earns received amount minus return.
- Mock video tells: smile duration, return delay before generous repayments.

## Commands
- Train: `python -m trust_game.train`
- Eval: `python -m trust_game.train --train-episodes 0 --eval-episodes 1000 --video false`
- Visualize: `python -m trust_game.train --seeds 1 --train-episodes 50 --eval-episodes 20 --video true`

## Outputs
- Default output location: `trust_game/outputs/`
- Override with: `--output-dir /absolute/path`

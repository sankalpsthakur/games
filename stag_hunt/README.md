# Stag Hunt

## Game Spec
- State: partner reliability belief, coordination history, and current round.
- Action: choose Stag (risky cooperative) or Hare (safe solo).
- Reward: both get high payoff on Stag/Stag; mismatch rewards Hare hunter only.
- Mock video tells: confident nod before Stag, hesitation frames before Hare.

## Commands
- Train: `python -m stag_hunt.train`
- Eval: `python -m stag_hunt.train --train-episodes 0 --eval-episodes 1000 --video false`
- Visualize: `python -m stag_hunt.train --seeds 1 --train-episodes 50 --eval-episodes 20 --video true`

## Outputs
- Default output location: `stag_hunt/outputs/`
- Override with: `--output-dir /absolute/path`

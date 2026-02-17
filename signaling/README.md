# Signaling

## Game Spec
- State: sender private type, receiver prior belief, and prior signal history.
- Action: sender chooses signal intensity; receiver chooses trust or reject.
- Reward: receiver gains on correct trust decisions; sender gains from accepted signals minus signal cost.
- Mock video tells: signal latency, voice wobble on deceptive signals.

## Commands
- Train: `python -m signaling.train`
- Eval: `python -m signaling.train --train-episodes 0 --eval-episodes 1000 --video false`
- Visualize: `python -m signaling.train --seeds 1 --train-episodes 50 --eval-episodes 20 --video true`

## Outputs
- Default output location: `signaling/outputs/`
- Override with: `--output-dir /absolute/path`

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

## Results (Multi-Method Framework)

30 seeds, 280 train steps, 140 eval steps.

| Algorithm | Mean | SD | SE | 95% CI | p vs zero |
|---|---:|---:|---:|---|---:|
| PPO | -0.1862 | 0.3781 | 0.0690 | [-0.3216, -0.0509] | 6.98e-03 |
| A2C | -0.1787 | 0.3212 | 0.0586 | [-0.2937, -0.0638] | 2.31e-03 |
| **DQN** | **-0.1508** | 0.3444 | 0.0629 | [-0.2740, -0.0275] | 1.65e-02 |

Best algorithm: **DQN** (mean -0.1508). Mildly negative; closest to zero among negative-reward games. Pairwise differences not significant.

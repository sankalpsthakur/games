# Adaptive Chess (Minimal)

This folder contains a minimal adaptive chess prototype:
- trains an opponent move model from simulated games
- uses mock facial-expression state variables every 10 seconds
- predicts opponent next moves from profile + tells
- prints shortest mating line under expected-opponent assumptions at each state

## File
- `adaptive_chess.py`

## Run
Use the existing local venv from poker:

```bash
/Users/sankalp/Projects/experiment/games/poker/.venv/bin/python /Users/sankalp/Projects/experiment/games/chess/adaptive_chess.py --profile tactical --train-games 300 --max-train-plies 80 --max-demo-plies 40 --max-mate-plies 6 --seed 42
```

## Profiles
- `balanced`
- `tactical`
- `solid`
- `blitzer`
- `endgame`

## Notes
- This is intentionally simple and lightweight.
- The "shortest mating sequence" is searched up to `--max-mate-plies` using deterministic expected opponent moves from the adaptive model.
- Real video feed integration is not wired yet; mock tells are used.

## Results (Multi-Method Framework)

30 seeds, 280 train steps, 140 eval steps.

| Algorithm | Mean | SD | SE | 95% CI | p vs zero |
|---|---:|---:|---:|---|---:|
| PPO | -0.3440 | 0.2358 | 0.0431 | [-0.4284, -0.2596] | 1.33e-15 |
| A2C | -0.3255 | 0.2437 | 0.0445 | [-0.4127, -0.2383] | 2.54e-13 |
| **DQN** | **-0.3123** | 0.2333 | 0.0426 | [-0.3958, -0.2288] | 2.28e-13 |

Best algorithm: **DQN** (mean -0.3123). All algorithms produce significantly negative reward. PPO vs DQN pairwise difference is marginally significant (p=0.049).

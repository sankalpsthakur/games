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

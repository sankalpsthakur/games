# Shared Multi-Method Simulation Framework

Lightweight deterministic framework for cross-game method simulation with:
- methods: `PPO`, `A2C`, `DQN` (no SB3 dependency)
- state: `base_game_state + 8-dim mock video tells (10s cadence) + opponent profile traits`
- evaluation: multi-seed significance protocol with paired method comparisons
- outputs: per-game `metrics.json` and required plots under `runs/<game>/`

## Files
- `opponent_intel.py`: opponent trait profiles and deterministic profile selection
- `mock_video_feed.py`: 8-dim tell generator sampled every 10 seconds
- `game_specs.py`: game registry and missed-game list
- `methods.py`: lightweight strategy learners (`PPO`, `A2C`, `DQN`)
- `eval_harness.py`: simulator + significance protocol (mean/sd/se/95% CI/z/p)
- `visualize.py`: `reward_ci.png`, `method_comparison.png`, `tell_sensitivity.png`
- `train_multi_method.py`: single-game CLI
- `run_all_missed_games.py`: parallel-safe multi-game orchestrator + summary reports

## Single Game
```bash
python -m shared.train_multi_method \
  --game tenders \
  --seeds 30 \
  --seed-start 1 \
  --train-steps 280 \
  --eval-steps 140 \
  --algorithms PPO,A2C,DQN \
  --output-root runs
```

Outputs:
- `runs/tenders/metrics.json`
- `runs/tenders/reward_ci.png`
- `runs/tenders/method_comparison.png`
- `runs/tenders/tell_sensitivity.png`

## All Missed Games
```bash
python -m shared.run_all_missed_games \
  --seeds 30 \
  --train-steps 280 \
  --eval-steps 140 \
  --jobs 3 \
  --output-root runs
```

Outputs:
- `runs/summary_report.json`
- `runs/summary_report.md`
- per-game `runs/<game>/...`

## Determinism Notes
- deterministic profile selection and tell generation per `(game, seed)`
- deterministic simulator dynamics and noise streams
- deterministic method initialization per `(game, method, seed)`

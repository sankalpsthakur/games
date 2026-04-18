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

## Generate PDF Report
```bash
python -m shared.generate_report_pdf --output-root runs
```

Produces a single PDF concatenating all per-game plots, the summary table, and the gap analysis.

## Results (All 18 Games)

Configuration: 30 seeds, 280 train steps, 140 eval steps, PPO + A2C + DQN, tell scales 0.0 / 0.5 / 1.0 / 1.5

| Game | Best Algo | Mean Reward | p-value |
|---|---|---:|---:|
| iterated_prisoners_dilemma | PPO | +0.4540 | 1.32e-06 |
| hawk_dove | DQN | +0.3156 | 3.51e-05 |
| stag_hunt | PPO | +0.2959 | 1.33e-03 |
| signaling | DQN | -0.1508 | 1.65e-02 |
| crisis_negotiations | A2C | -0.1977 | 5.37e-03 |
| beer_distribution | DQN | -0.2242 | 7.58e-04 |
| principal_agent | DQN | -0.2486 | 1.38e-05 |
| market_for_lemons | DQN | -0.2581 | 3.13e-05 |
| poker | A2C | -0.2820 | 1.13e-05 |
| chess | DQN | -0.3123 | 2.28e-13 |
| trust_game | A2C | -0.3141 | 2.33e-07 |
| auctions | PPO | -0.3370 | 5.70e-10 |
| cournot | A2C | -0.3401 | 8.30e-11 |
| interrogation | DQN | -0.3478 | 8.08e-11 |
| ultimatum | A2C | -0.3515 | 7.79e-09 |
| tenders | DQN | -0.3586 | 0.00e+00 |
| bertrand | DQN | -0.3741 | 9.06e-13 |
| public_goods | PPO | -0.3819 | 2.29e-09 |

Algorithm wins: DQN=9, A2C=5, PPO=4. Only 3/18 games achieve positive reward.

## Determinism Notes
- deterministic profile selection and tell generation per `(game, seed)`
- deterministic simulator dynamics and noise streams
- deterministic method initialization per `(game, method, seed)`

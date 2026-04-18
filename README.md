# Multi-Method Game Strategy Simulation Framework

A deterministic simulation framework that trains and evaluates reinforcement learning strategies (PPO, A2C, DQN) across 18 game-theoretic environments. Each game incorporates mock video tells (8-dim behavioral signals) and opponent profiling to study how additional information channels affect strategic decision-making.

## Architecture

```
game_specs       Define state/action/reward for each game (shared/game_specs.py)
    |
    v
simulator        Deterministic environment dynamics + noise streams (shared/eval_harness.py)
    |
    v
methods          PPO, A2C, DQN lightweight learners (shared/methods.py)
    |
    v
eval_harness     Multi-seed significance protocol: mean/sd/se/95% CI/z/p (shared/eval_harness.py)
    |
    v
visualize        reward_ci.png, method_comparison.png, tell_sensitivity.png (shared/visualize.py)
    |
    v
reports          summary_report.md, per-game metrics.json, gap_analysis.md, PDF
```

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

## Algorithm Comparison

| Algorithm | Wins | Games |
|---|---:|---|
| DQN | 9 | hawk_dove, signaling, beer_distribution, principal_agent, market_for_lemons, chess, interrogation, tenders, bertrand |
| A2C | 5 | crisis_negotiations, poker, trust_game, cournot, ultimatum |
| PPO | 4 | iterated_prisoners_dilemma, stag_hunt, auctions, public_goods |

DQN leads in raw win count, though pairwise differences between algorithms are generally small and often not statistically significant.

## Game Categories

**Positive reward (3/18):** iterated_prisoners_dilemma, hawk_dove, stag_hunt -- all are 2-action coordination games where cooperative equilibria are reachable.

**Negative reward (15/18):** The remaining games produce negative mean rewards. These span multi-action settings (auctions, poker, chess), information asymmetry games (signaling, market_for_lemons, principal_agent), and zero-sum or adversarial games (bertrand, cournot, tenders).

## How to Run

### Single game
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

### All 18 games
```bash
python -m shared.run_all_missed_games \
  --seeds 30 \
  --train-steps 280 \
  --eval-steps 140 \
  --jobs 3 \
  --output-root runs
```

### Generate PDF report
```bash
python -m shared.generate_report_pdf --output-root runs
```

## Documentation

- [Framework details](shared/README.md) -- shared module documentation
- [Gap analysis](runs/gap_analysis.md) -- honest assessment of results and limitations
- [Summary report](runs/summary_report.md) -- tabular overview of all 18 games

## Project Structure

```
games/
  shared/                  Shared framework (methods, eval harness, visualize)
  runs/                    All simulation outputs, summary reports, gap analysis
    <game>/metrics.json    Per-game detailed results
    <game>/*.png           Per-game plots
    summary_report.md      Tabular summary
    gap_analysis.md        Honest gap analysis
  auctions/                Auction game implementation
  beer_distribution/       Beer distribution (supply chain) game
  bertrand/                Bertrand price competition
  chess/                   Adaptive chess prototype
  cournot/                 Cournot quantity competition
  crisis_negotiations/     Crisis negotiation game
  hawk_dove/               Hawk-Dove coordination game
  interrogation/           Interrogation game
  iterated_prisoners_dilemma/  Iterated Prisoner's Dilemma
  market_for_lemons/       Market for lemons (adverse selection)
  poker/                   Poker (Texas Hold'em variant)
  principal_agent/         Principal-agent contract game
  public_goods/            Public goods contribution game
  signaling/               Signaling game
  stag_hunt/               Stag Hunt coordination game
  tenders/                 Competitive tendering game
  trust_game/              Trust (investment) game
  ultimatum/               Ultimatum bargaining game
```

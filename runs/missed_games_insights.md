# Missed Games Insights

## Aggregate
- Algorithm win counts: PPO=5, A2C=6, DQN=5
- Games with all methods positive: hawk_dove, iterated_prisoners_dilemma, stag_hunt
- Games with all methods negative: auctions, beer_distribution, bertrand, cournot, crisis_negotiations, interrogation, market_for_lemons, principal_agent, public_goods, signaling, tenders, trust_game, ultimatum
- Strongest tell-sensitivity delta (1.5-0.0): ultimatum A2C delta=-0.2188 (-0.4039->-0.6227)

## Per Game

| Game | Best | Mean | 95% CI | p(vs 0) | Pairwise p<0.05 |
|---|---|---:|---|---:|---|
| auctions | PPO | -0.3361 | [-0.4408, -0.2314] | 3.14e-10 | - |
| beer_distribution | DQN | -0.2370 | [-0.3539, -0.1201] | 7.06e-05 | - |
| bertrand | PPO | -0.3720 | [-0.4796, -0.2644] | 1.24e-11 | - |
| cournot | A2C | -0.3435 | [-0.4483, -0.2387] | 1.33e-10 | - |
| crisis_negotiations | A2C | -0.2076 | [-0.3423, -0.0729] | 0.00252 | - |
| hawk_dove | A2C | +0.3265 | [+0.1610, +0.4921] | 0.000111 | PPO_minus_A2C |
| interrogation | A2C | -0.3491 | [-0.4624, -0.2358] | 1.54e-09 | - |
| iterated_prisoners_dilemma | PPO | +0.4419 | [+0.2618, +0.6221] | 1.52e-06 | - |
| market_for_lemons | DQN | -0.2646 | [-0.3871, -0.1421] | 2.29e-05 | - |
| principal_agent | PPO | -0.2390 | [-0.3512, -0.1269] | 2.94e-05 | - |
| public_goods | PPO | -0.3808 | [-0.5070, -0.2545] | 3.4e-09 | - |
| signaling | DQN | -0.1611 | [-0.2836, -0.0387] | 0.00991 | - |
| stag_hunt | DQN | +0.3327 | [+0.2003, +0.4652] | 8.52e-07 | - |
| tenders | DQN | -0.3519 | [-0.4177, -0.2862] | 0 | - |
| trust_game | A2C | -0.3168 | [-0.4363, -0.1973] | 2.04e-07 | - |
| ultimatum | A2C | -0.3550 | [-0.4727, -0.2374] | 3.31e-09 | - |

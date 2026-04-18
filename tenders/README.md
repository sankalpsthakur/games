# Tenders

## Game Spec
- State: project value, private cost estimate, and competitor win-rate belief.
- Action: submit bid amount or skip tender.
- Reward: winner gets project margin; losers get zero.
- Mock video tells: bid-submit delay, confidence smile before aggressive bids.

## Commands
- Train: `python -m tenders.train`
- Eval: `python -m tenders.train --train-episodes 0 --eval-episodes 1000 --video false`
- Visualize: `python -m tenders.train --seeds 1 --train-episodes 50 --eval-episodes 20 --video true`

## Outputs
- Default output location: `tenders/outputs/`
- Override with: `--output-dir /absolute/path`

## Results (Multi-Method Framework)

30 seeds, 280 train steps, 140 eval steps.

| Algorithm | Mean | SD | SE | 95% CI | p vs zero |
|---|---:|---:|---:|---|---:|
| PPO | -0.4075 | 0.1984 | 0.0362 | [-0.4785, -0.3365] | 0.00e+00 |
| A2C | -0.3904 | 0.1837 | 0.0335 | [-0.4562, -0.3247] | 0.00e+00 |
| **DQN** | **-0.3586** | 0.1866 | 0.0341 | [-0.4254, -0.2919] | 0.00e+00 |

Best algorithm: **DQN** (mean -0.3586). All algorithms strongly negative with low variance. PPO vs DQN pairwise difference marginally significant (p=0.044).

# Principal Agent

## Game Spec
- State: contract terms, effort proxy, and shirking prior.
- Action: principal sets wage/bonus; agent chooses effort level.
- Reward: principal gets output minus compensation; agent gets compensation minus effort cost.
- Mock video tells: compliance glance, fatigue marker after high effort.

## Commands
- Train: `python -m principal_agent.train`
- Eval: `python -m principal_agent.train --train-episodes 0 --eval-episodes 1000 --video false`
- Visualize: `python -m principal_agent.train --seeds 1 --train-episodes 50 --eval-episodes 20 --video true`

## Outputs
- Default output location: `principal_agent/outputs/`
- Override with: `--output-dir /absolute/path`

# Gap Analysis: Multi-Method Game Strategy Simulation

**Date:** 2026-02-18
**Scope:** 18 games, 3 algorithms (PPO, A2C, DQN), 30 seeds, 280 train steps, 140 eval steps

---

## 1. Performance Summary

The results are poor. Only **3 out of 18 games** produce positive mean reward. The remaining 15 games yield significantly negative rewards, confirmed at high statistical confidence.

| Game | Best Algo | Mean Reward | p-value | Verdict |
|---|---|---:|---:|---|
| iterated_prisoners_dilemma | PPO | **+0.4540** | 1.32e-06 | Positive |
| hawk_dove | DQN | **+0.3156** | 3.51e-05 | Positive |
| stag_hunt | PPO | **+0.2959** | 0.00133 | Positive |
| signaling | DQN | -0.1508 | 0.0165 | Negative |
| crisis_negotiations | A2C | -0.1977 | 0.00537 | Negative |
| beer_distribution | DQN | -0.2242 | 0.000758 | Negative |
| principal_agent | DQN | -0.2486 | 1.38e-05 | Negative |
| market_for_lemons | DQN | -0.2581 | 3.13e-05 | Negative |
| poker | A2C | -0.2820 | 1.13e-05 | Negative |
| chess | DQN | -0.3123 | 2.28e-13 | Negative |
| trust_game | A2C | -0.3141 | 2.33e-07 | Negative |
| auctions | PPO | -0.3370 | 5.70e-10 | Negative |
| cournot | A2C | -0.3401 | 8.30e-11 | Negative |
| interrogation | DQN | -0.3478 | 8.08e-11 | Negative |
| ultimatum | A2C | -0.3515 | 7.79e-09 | Negative |
| tenders | DQN | -0.3586 | 0.00e+00 | Negative |
| bertrand | DQN | -0.3741 | 9.06e-13 | Negative |
| public_goods | PPO | -0.3819 | 2.29e-09 | Negative |

**Algorithm win counts:** PPO=4, DQN=9, A2C=5. DQN wins the most games, but this is misleading -- it simply loses slightly less badly in most cases.

**Honest bottom line:** The system does not work for the majority of games. Positive results on 3 games are real but likely explained by those games being trivially simple (see Section 6). For 15/18 games the agent performs worse than doing nothing.

---

## 2. Method Limitations

All three "algorithms" are severely stripped down relative to their real counterparts.

### 2.1 Linear Function Approximation Only

Every learner uses a single linear layer (`W @ state + b`) with no hidden layers, no activation functions, and no neural network of any kind:

- **DQN** (`methods.py:69-70`): `q_w` is `(n_actions, state_dim)`, `q_b` is `(n_actions,)`. Q-values are `q_w @ state + q_b`.
- **A2C** (`methods.py:131-134`): Actor is `actor_w @ state + actor_b`; critic is `value_w @ state + value_b`. Both linear.
- **PPO** (`methods.py:178`): Inherits from A2C; same linear architecture. The clip mechanism modulates the gradient scale but still operates on a linear policy.

A linear function approximator cannot learn nonlinear decision boundaries. For any game where the optimal action depends on interactions between state features (which is most games), the learner is fundamentally incapable of representing the correct policy. This is the single biggest limitation.

### 2.2 DQN Missing Core Components

The DQN implementation (`methods.py:59-121`) is missing two components that made the original DQN work:

- **No experience replay buffer.** The agent learns from each transition once, immediately, and discards it. This means highly correlated sequential samples, which destabilizes learning.
- **No target network.** The same weights are used for both Q-value estimation and target computation. The target shifts with every update, creating a moving-target problem.

Without these, this is not really DQN -- it is online linear Q-learning with epsilon-greedy exploration.

### 2.3 PPO Clip Is Approximate

PPO (`methods.py:186-224`) computes a ratio `new_prob / old_prob` and clips it, but the underlying policy update is still a scaled linear gradient step. Real PPO runs multiple epochs of minibatch updates on a collected trajectory; this implementation does a single online update per step. The clip mechanism here constrains the gradient magnitude but does not provide the trust-region-like guarantees of actual PPO.

### 2.4 A2C Missing Entropy Bonus

The A2C implementation (`methods.py:123-175`) has no entropy regularization term. Entropy bonuses are critical for preventing premature convergence to deterministic policies. Without it, A2C quickly collapses to greedy behavior before exploring the action space sufficiently.

### 2.5 Training Budget Is Extremely Short

All games train for exactly **280 steps**. For context:
- The original DQN paper trained for 50 million frames.
- Standard RL benchmarks use thousands to millions of environment steps.
- 280 steps with a linear model means the weight matrix receives 280 gradient updates total. For a game like tenders with 12 actions and 26-dimensional state (13 base + 8 tell + 5 profile), that is 280 updates to a (12, 26) weight matrix -- roughly 0.9 updates per parameter on average.

This is not enough training for any method to learn meaningful behavior on complex tasks.

---

## 3. Simulation Limitations

### 3.1 Reward Function Is Classification-Based, Not Game-Theoretic

The environment (`eval_harness.py:189-193`) computes reward as:

```python
classification_reward = 1.0 if action == optimal_action else (-0.35 - 0.15 * margin)
signal_reward = 0.25 * expected[action]
reward = classification_reward + signal_reward + noise[action]
```

This is dense oracle-based supervision: the agent gets +1 for matching the oracle's preferred action and a penalty proportional to the regret margin otherwise. This is closer to supervised classification than reinforcement learning. Real game-theoretic payoffs (e.g., Nash equilibrium rewards, actual prisoner's dilemma payoff matrices) are not used.

The "optimal action" is defined by `np.argmax(expected_rewards)`, where `expected_rewards` is itself a synthetic function of the state, action weights, tell features, and profile. The agent is learning to imitate a randomly-seeded linear oracle, not to play a real game.

### 3.2 Environments Are Synthetic and Parametric

The `LightweightGameSimulator` (`eval_harness.py:91-205`) generates states and rewards entirely from `game_specs.py` parameters. Key observations:

- **Base states** are RNG-generated with sinusoidal/cosinusoidal modulation (`eval_harness.py:128-139`). They do not represent real game states (board positions, card distributions, auction bids, etc.).
- **Tell features** are synthetic from `MockVideoFeed` (`mock_video_feed.py:40-115`), producing 8-dimensional vectors (gaze_aversion, blink_rate, jaw_tension, etc.) that are deterministic functions of profile traits and step index. These are not actual behavioral signals.
- **The action-reward mapping** uses a random weight matrix (`eval_harness.py:113-114`) scaled by difficulty, tell_weight, and profile_weight. The "game" is: match the argmax of a noisy linear function. Every game uses the same structure; they differ only in dimensionality, noise level, and scaling parameters.

This means the 18 "games" are 18 parameterizations of the same synthetic environment, not 18 different game-theoretic settings.

### 3.3 No Opponent Modeling or Multi-Agent Interaction

All evaluations are single-agent against a static environment. There is:

- No opponent that adapts, learns, or responds to the agent's strategy.
- No self-play or population-based training.
- No multi-agent dynamics whatsoever.

This is especially problematic for games where opponent interaction is definitional (prisoner's dilemma, hawk-dove, stag hunt, negotiation games, auctions). The agent is not "playing" these games in any meaningful sense.

### 3.4 Opponent Profiles Are Fixed and Deterministic

The `choose_profile` function (`opponent_intel.py:59-63`) deterministically selects from 5 fixed profiles (balanced, aggressive, tight, tricky, composed) based on a hash of game name and seed. The profile does not change during an episode. A real opponent would vary behavior dynamically.

---

## 4. Statistical Concerns

### 4.1 Sample Size (30 Seeds)

With 30 seeds per game per algorithm, the 95% confidence intervals are moderately wide. Representative examples from the data:

| Game | Algo | Mean | CI95 Width | SD |
|---|---|---:|---:|---:|
| iterated_prisoners_dilemma | PPO | +0.454 | 0.368 | 0.514 |
| hawk_dove | DQN | +0.316 | 0.299 | 0.418 |
| stag_hunt | A2C | +0.279 | 0.361 | 0.507 |
| bertrand | DQN | -0.374 | 0.205 | 0.287 |
| tenders | DQN | -0.359 | 0.134 | 0.187 |

For the 3 positive games, CI widths range from 0.30 to 0.37, meaning the true mean could be anywhere from modestly positive to strongly positive. This is adequate for directional conclusions but not for precise effect-size estimation. For the negative games, the CIs are tighter (0.13-0.25) because SDs are smaller, making the "significantly negative" conclusion robust.

30 seeds is sufficient to detect the direction of effect but insufficient for fine-grained algorithm comparisons.

### 4.2 Tiny P-Values Do Not Mean Good Performance

Most p-values are astronomically small (e.g., p=0 for tenders, p=9e-13 for bertrand). This is a statistical trap: these p-values test "is the mean different from zero?" not "is the performance good?" The answer is yes, the performance is reliably and significantly **bad**. The small p-values simply confirm that the negative bias is real and reproducible, not random.

For the 3 positive games, the p-values (1.3e-6, 3.5e-5, 1.3e-3) confirm the positive bias is also real, but the effect sizes (+0.29 to +0.45) are modest.

### 4.3 No Multiple Testing Correction

18 games are tested simultaneously. Without Bonferroni or FDR correction:
- Bonferroni-adjusted alpha at 0.05: 0.05/18 = 0.00278
- Under Bonferroni, signaling (p=0.0165) and crisis_negotiations (p=0.00537) would lose significance.
- The 3 positive games all survive Bonferroni correction.
- The strongly negative games (p < 1e-5) all survive any reasonable correction.

The lack of correction does not change the main conclusions but is a methodological gap.

### 4.4 Pairwise Algorithm Comparisons Are Uninformative

Looking at pairwise differences across the metrics data:

| Game | Comparison | Mean Diff | p-value |
|---|---|---:|---:|
| iterated_prisoners_dilemma | PPO - DQN | +0.0069 | 0.691 |
| iterated_prisoners_dilemma | PPO - A2C | +0.0017 | 0.856 |
| hawk_dove | PPO - DQN | -0.0223 | 0.599 |
| bertrand | PPO - DQN | -0.0079 | 0.810 |
| tenders | PPO - DQN | -0.0489 | 0.044 |

Almost all pairwise p-values are non-significant (p > 0.05). The three algorithms perform nearly identically on most games. This makes sense: they are all linear models with similar capacity. There is no meaningful algorithm selection happening; the "best" algorithm per game is determined by noise, not by genuine algorithmic advantage.

---

## 5. Missing Features

### 5.1 No Hyperparameter Tuning

All games use identical hyperparameters hardcoded in `methods.py`:

| Parameter | DQN | A2C | PPO |
|---|---:|---:|---:|
| Learning rate | 0.035 | 0.016 (actor), 0.024 (critic) | 0.014 (actor), 0.022 (critic) |
| Gamma | 0.96 | 0.96 | 0.96 |
| Epsilon start/end | 0.22/0.02 | -- | -- |
| Clip ratio | -- | -- | 0.20 |
| Temperature | -- | 1.0 | 1.0 |

No game-specific tuning is performed. Games with 2 actions (hawk_dove, stag_hunt, IPD) and games with 12 actions (tenders, interrogation) use identical learning rates and exploration parameters. This one-size-fits-all approach almost certainly hurts performance on games at the extremes.

### 5.2 No Random Baseline

There is no comparison to a uniform-random policy. A random agent choosing actions uniformly would get the classification reward (+1) with probability 1/n_actions per step. For 2-action games, random gets +1 half the time and roughly -0.35 half the time, yielding ~+0.325 expected reward. This means **the positive results on 2-action games may not even beat random**. Without a random baseline, we cannot assess whether the learners are doing anything useful.

### 5.3 No Oracle Baseline

There is no comparison to an agent that always picks the optimal action. Since the environment provides oracle information (`info["optimal_action"]`), computing the oracle upper bound is trivial. Without it, we cannot assess what fraction of possible performance the learners capture.

### 5.4 No Curriculum Learning or Staged Training

All 280 training steps use the same environment difficulty. No curriculum (easy-to-hard), no warm-start, no staged learning.

### 5.5 No Architecture Search

The linear architecture was chosen, not validated. No comparison to:
- 1-hidden-layer MLP
- Tabular methods (for small action spaces)
- Simple lookup tables
- Nearest-neighbor approaches

### 5.6 No Model Selection Framework

The "best algorithm" is selected post-hoc by comparing means. There is no cross-validation, no holdout set, no model selection criterion. The 30 evaluation seeds are used both for selection and for reporting, which inflates the best-of-3 mean through selection bias.

---

## 6. Why Only 3 Games Succeed

The 3 positive games share critical properties:

| Game | n_actions | difficulty | base_state_dim | reward_noise |
|---|---:|---:|---:|---:|
| iterated_prisoners_dilemma | **2** | **0.56** | 9 | 0.14 |
| hawk_dove | **2** | **0.57** | 8 | 0.15 |
| stag_hunt | **2** | **0.58** | 8 | 0.14 |

Compare to the worst-performing games:

| Game | n_actions | difficulty | base_state_dim | reward_noise |
|---|---:|---:|---:|---:|
| tenders | 12 | 0.70 | 13 | 0.16 |
| bertrand | 9 | 0.64 | 10 | 0.15 |
| public_goods | 11 | 0.60 | 10 | 0.15 |
| interrogation | 12 | 0.71 | 12 | 0.18 |

The pattern is clear:

1. **All 3 successes are 2-action games.** With only 2 actions, the classification reward gives +1 on the correct action and ~-0.35 on the wrong action. A linear model with 280 training steps has a reasonable chance of learning which side of a hyperplane each state falls on. For 9-12 action games, the correct action is a 1-of-N classification problem where the linear model must partition a high-dimensional space into many more regions.

2. **The 3 successes have the lowest difficulty scores** (0.56-0.58). The difficulty parameter directly scales the action weights (`eval_harness.py:116`): higher difficulty means the signal distinguishing optimal from suboptimal actions is weaker. Lower difficulty means clearer signal.

3. **Smaller state dimension.** 8-9 base dimensions vs. 10-14 for harder games. Fewer parameters to learn with the same 280-step budget.

4. **Random baseline concern (revisited).** For 2-action games, a random agent picks optimally 50% of the time, yielding an expected reward around +0.325. The best results (+0.454, +0.316, +0.296) are in the same ballpark. It is plausible that the positive-game learners are performing only marginally above random chance.

---

## 7. Tell Sensitivity Analysis

The sensitivity sweep (tell_scales: 0.0, 0.5, 1.0, 1.5) reveals:

**Positive games (IPD, stag_hunt):** Performance is positive across all tell scales, including tell_scale=0.0 (tells zeroed out). For IPD A2C, tell_scale=0.0 gives mean=+0.234 and tell_scale=1.0 gives mean=+0.450. The tell features provide some signal but performance is positive even without them, suggesting the agents are mostly learning from base state and profile features, not from tells.

**Negative games (bertrand, poker):** Performance is negative at all tell scales. For bertrand A2C: tell_scale=0.0 gives mean=-0.354, tell_scale=1.0 gives mean=-0.385. Varying tell intensity does not rescue performance. For poker A2C: tell_scale=0.0 gives mean=-0.287, tell_scale=0.5 gives mean=-0.145, tell_scale=1.0 gives mean=-0.281. The non-monotonic pattern suggests the agents are not learning meaningful tell-to-action mappings.

Overall, tell sensitivity shows that tells are not a primary driver of performance in either direction.

---

## 8. Recommendations

### Immediate (Low Effort)

1. **Add random and oracle baselines.** Compute expected reward for uniform-random and always-optimal policies. Report all results as "reward above random" and "fraction of oracle performance." This single change would dramatically clarify whether the learners are doing anything useful.

2. **Apply Bonferroni correction** to reported p-values. Report both raw and adjusted values.

3. **Increase training budget.** Even 2,800 steps (10x current) with linear models would provide a fairer test. The current 280 steps is insufficient for any nontrivial learning.

### Medium Term (Moderate Effort)

4. **Add at least one hidden layer** (e.g., 64- or 128-unit MLP with ReLU). This addresses the fundamental representational bottleneck. A single hidden layer would allow learning nonlinear decision boundaries.

5. **Implement proper DQN components:** experience replay buffer (even a small one, 1000-5000 transitions) and a target network (update every 50-100 steps). These are essential for stable Q-learning.

6. **Add entropy bonus to A2C.** Standard coefficient of 0.01-0.05 on the entropy of the policy distribution.

7. **Per-game hyperparameter tuning.** At minimum, grid-search over learning rate and exploration parameters per game category (2-action vs. many-action).

### Longer Term (Significant Effort)

8. **Replace synthetic environments with real game-theoretic payoff structures.** Use actual payoff matrices for matrix games, actual auction mechanisms for auction games, etc. The current `LightweightGameSimulator` makes all games structurally identical.

9. **Implement multi-agent training.** Self-play or population-based training for games where opponent interaction matters.

10. **Add opponent adaptation.** Dynamic opponents that adjust strategy in response to the agent's behavior, rather than fixed profiles.

11. **Curriculum learning.** Start with low difficulty and ramp up. Start with fewer actions and add complexity.

12. **Proper model selection.** Cross-validation or separate holdout seeds for algorithm selection vs. performance reporting.

---

## 9. Summary Verdict

This is a simulation framework, not a competitive game-playing system. The 83% failure rate (15/18 negative) is a direct consequence of fundamental design choices: linear function approximation, minimal training budget, synthetic environments, and stripped-down algorithm implementations. The 3 successes on 2-action/low-difficulty games may not meaningfully exceed a random baseline. The framework demonstrates the mechanics of running multi-method RL experiments but does not yet produce agents that play games effectively.

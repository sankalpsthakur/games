# Task ID: 6

**Title:** Upgrade PPO implementation for multi-classified games

**Status:** pending

**Dependencies:** 3, 5

**Priority:** high

**Description:** Enhance PPO in `shared/methods.py` with rollout buffer semantics, multi-epoch updates, KL/value clipping diagnostics, and stable advantage normalization.

**Details:**

Refactor PPO into explicit buffer collection + update stages. Add GAE options, clipping safeguards, entropy logging, and guardrails for batch/trajectory length mismatches. Include support for variable action cardinalities from game specs.

**Test Strategy:**

Train PPO for a short horizon on a toy game and assert loss curves are bounded, entropy stays finite, and update step changes policy only when advantage estimates are computable.

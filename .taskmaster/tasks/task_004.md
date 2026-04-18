# Task ID: 4

**Title:** Add baseline suite module and control variants

**Status:** pending

**Dependencies:** 1

**Priority:** high

**Description:** Create `shared/baselines.py` with random, oracle, greedy/no-learning, tell-disabled, and profile-disabled controls plus ablation toggles.

**Details:**

Implement baseline classes with a unified policy interface used by training and evaluation loops. Add switch matrix support so each game run can compare learned agent against all baselines under identical seeds and budgets.

**Test Strategy:**

Smoke test that each baseline returns valid action vectors and consumes identical feature shapes as the learner policies for at least two games. Verify toggles correctly disable tell messages, profile drift, and adaptation.

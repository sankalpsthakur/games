# Task ID: 5

**Title:** Extend game metadata and class tags for tuning strategy

**Status:** pending

**Dependencies:** 2

**Priority:** medium

**Description:** Update `shared/game_specs.py` with reward mode, game class labels, equilibrium targets, and opponent defaults used by downstream schedulers.

**Details:**

Add fields for action space type, equilibrium hypothesis, strategic class (`binary`, `low_entropy`, `multi_action`, `adversarial`), and terminal diagnostics hooks. Export utility methods mapping game to recommended learning method, LR schedule, and rollout depth.

**Test Strategy:**

Add unit tests that every registered game has a valid class tag and that every runner query of game class produces deterministic, typed config values.

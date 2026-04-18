# Task ID: 9

**Title:** Add architecture-aware per-game training budgets and method-routing

**Status:** pending

**Dependencies:** 5, 6, 7, 8

**Priority:** high

**Description:** Wire game class tags into `shared/train_multi_method.py` and runner plumbing so method/hyperparameters are chosen per game and per evaluation block.

**Details:**

Build a declarative routing table that maps game class and sample regime to method family, learning-rate envelope, episode budget, entropy floor, and replay/rollout depth. Add CLI-level overrides with safe validation and run manifest capture.

**Test Strategy:**

Run a matrix config through the trainer and assert each of 4 sample game classes gets its intended method/hyperparameter set and produces per-game budgets in manifests.

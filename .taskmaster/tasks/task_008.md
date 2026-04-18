# Task ID: 8

**Title:** Upgrade DQN for robust discrete-action studies

**Status:** pending

**Dependencies:** 3, 5

**Priority:** medium

**Description:** Enhance `shared/methods.py` DQN path with replay memory, target network sync, and optional Double/Dueling variants.

**Details:**

Introduce prioritized/uniform replay abstraction, periodic target update schedule, epsilon annealing with lower bounds, and optional Double/Dueling heads. Add action-availability masking for invalid moves from game specs.

**Test Strategy:**

Run scripted batch over known transitions and assert target network lag is respected, replay samples shape-correct, and epsilon schedule decays monotonically.

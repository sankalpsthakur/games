# Task ID: 7

**Title:** Upgrade A2C implementation with stability controls

**Status:** pending

**Dependencies:** 3, 5

**Priority:** medium

**Description:** Harden A2C in `shared/methods.py` with entropy bonus, value clipping, reward normalization, and rollout sanity checks.

**Details:**

Add entropy regularization, orthogonalized advantage normalization, and clipped value loss. Add runtime assertions for NaN/inf checks and gradient-norm reporting so training collapses are diagnosed quickly.

**Test Strategy:**

Run short training pass and validate no NaN in actor/critic outputs, entropy term contributes to loss, and episode return metrics remain finite over 500 steps.

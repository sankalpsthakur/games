# Task ID: 3

**Title:** Introduce explicit opponent API and adaptive policy strategies

**Status:** pending

**Dependencies:** 1, 2

**Priority:** high

**Description:** Add `shared/opponents.py` with self-play, adaptive, and static opponent profiles, then integrate it into environment execution paths.

**Details:**

Create a typed opponent interface with methods for profile update, belief update, mixed-policy sampling, and horizon-aware adaptation. Add baseline profiles: random, minimax-lite, exploitative best-response, and curriculum self-play. Add hooks for per-step opponent state persistence and deterministic seeding.

**Test Strategy:**

Simulate one small game with fixed seeds and assert opponent trajectories are reproducible and policy switches follow profile schedule files. Add contract tests for deterministic and adaptive branches.

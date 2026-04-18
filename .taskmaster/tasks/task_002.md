# Task ID: 2

**Title:** Replace synthetic oracle reward with explicit payoff logic

**Status:** pending

**Dependencies:** 1

**Priority:** high

**Description:** Refactor `shared/eval_harness.py` and related flow to compute game outcomes from explicit game-state payoffs instead of oracle-style shortcuts.

**Details:**

Define explicit per-game reward handlers using terminal state, move legality, and public-payoff semantics. Add a shared payoff contract so each game can inject utility, regret signals, and terminal diagnostics. Keep backward compatibility by retaining a compatibility shim behind a deprecated config flag.

**Test Strategy:**

Create deterministic fixtures for at least three representative games and assert rewards match manually computed payoff tables for fixed transcripts. Include failing oracle-path regression tests that confirm reward deltas are non-zero in relevant games.

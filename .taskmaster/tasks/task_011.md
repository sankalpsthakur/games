# Task ID: 11

**Title:** Add game-theoretic metrics: regret, exploitability, and welfare

**Status:** pending

**Dependencies:** 2, 10

**Priority:** high

**Description:** Create `shared/metrics.py` and expose regret/exploitability, welfare/fairness, and convergence-rate indicators for all reportable runs.

**Details:**

Compute per-episode and aggregate metrics across methods, including best-response regret gaps, empirical exploitability proxies, social welfare, and fairness dispersion. Include confidence-ready variance estimates and seed-level diagnostics.

**Test Strategy:**

Introduce controlled synthetic payoffs and verify each metric matches closed-form expectations for at least one cooperative and one competitive toy scenario.

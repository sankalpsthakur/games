# Task ID: 14

**Title:** Execute full 18-game smoke sweep with ablations

**Status:** pending

**Dependencies:** 9, 10, 11, 13

**Priority:** medium

**Description:** Run baseline + all algorithm suites over the full game portfolio with lower seeds, collecting reproducible diagnostics before full-scale final runs.

**Details:**

Automate matrix generation for ablations (oracle/no-oracle baseline, tell disabled, profile disabled, opponent static/adaptive), produce aggregated manifests, and check pipeline completes end-to-end across 18 games.

**Test Strategy:**

Execute pipeline with smoke seed budget and assert all 18 games emit manifest, metrics, and visualization files; fail pipeline if any game missing paired baseline metrics or has invalid metadata.

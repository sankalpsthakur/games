# Task ID: 13

**Title:** Enhance reporting and visualization with gap-ledger sections

**Status:** pending

**Dependencies:** 11, 12

**Priority:** high

**Description:** Update `shared/visualize.py` and report writers to emit full 18-game comparisons, confidence intervals, and explicit gap statements for low-confidence findings.

**Details:**

Add report sections for baseline parity checks, method ranking heatmaps, gap ledger entries per game (what insight is missing, what failed to converge), and confidence labeling for each conclusion. Surface invalid/missing-data warnings in output artifacts.

**Test Strategy:**

Generate a sample report and assert schema fields exist for all 18 games, include baseline diffs, CI spans, and a machine-readable gap ledger section in JSON/markdown.

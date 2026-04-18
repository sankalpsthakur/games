# Task ID: 12

**Title:** Build statistical protocol for paired, multi-comparison reporting

**Status:** pending

**Dependencies:** 10, 11

**Priority:** high

**Description:** Implement paired-seed comparisons, effect-size summaries, and multiple-testing corrections across method/game cells.

**Details:**

Add utilities for paired t/Wilcoxon where applicable, bootstrap CIs, and FDR/Bonferroni pathways. Ensure all outputs include sample count, seed pairing id, and adjusted p-values where declared.

**Test Strategy:**

Unit-test on synthetic Gaussian returns with known significance outcomes. Verify corrected p-values and effect sizes match known library values.

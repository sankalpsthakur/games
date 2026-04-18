# Task ID: 10

**Title:** Integrate opponent dynamics into training and eval loops

**Status:** pending

**Dependencies:** 3, 9

**Priority:** high

**Description:** Modify runners so adaptive opponent profiles and self-play populations are first-class citizens in each episode and ablation scenario.

**Details:**

Add lifecycle hooks for opponent initialization, profile drift, and stability gates. Persist opponent states to run manifests. Ensure tell/profile disabling options map cleanly to corresponding behavior without contaminating control studies.

**Test Strategy:**

Compare two runs with identical seeds but different opponent schedules and verify episode transcripts diverge only where expected. Add regression test for disabling profile adaptation in baseline conditions.

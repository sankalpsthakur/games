# Task ID: 1

**Title:** Establish simulation experiment ledger and baseline controls

**Status:** pending

**Dependencies:** None

**Priority:** high

**Description:** Create a reproducible run-manifest and experiment-control foundation for all future 18-game studies, including explicit schema versioning and seed bookkeeping.

**Details:**

Implement a shared artifact contract: manifest fields for repo commit, run_schema_version, game_id, method, baseline_family, opponent_profile, seed, hyperparameters, hardware/runtime metadata, and data quality flags. Add helpers to validate/write manifests so every run emits immutable JSONL artifacts. Stand up a baseline matrix scaffold and a single config entrypoint for smoke vs full-seed runs.

**Test Strategy:**

Run a dummy job and assert one manifest file exists per run with validated schema; verify seed list uniqueness and deterministic output filenames across reruns.

#!/usr/bin/env python3
"""Run multi-method poker significance protocol across many seeds.

This script trains/evaluates multiple algorithms on the realistic Hold'em setup
and computes per-profile and paired significance summaries.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from run_experiment import DEFAULT_PROFILES, run_experiment

DEFAULT_OUTPUT = Path(
    "/Users/sankalp/Projects/experiment/games/poker/realistic/results/significance_multi_method.json"
)


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def two_sided_p_from_z(z: float) -> float:
    return 2.0 * (1.0 - normal_cdf(abs(z)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-method 30-seed significance protocol")
    parser.add_argument("--algorithms", type=str, default="PPO,A2C,DQN")
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--train-profile", type=str, default="balanced")
    parser.add_argument("--train-steps", type=int, default=10_000)
    parser.add_argument("--eval-hands", type=int, default=200)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def stats_against_zero(values: List[float]) -> Dict[str, float]:
    n = len(values)
    mean = statistics.fmean(values) if n else float("nan")
    sd = statistics.stdev(values) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    ci_low = mean - 1.96 * se if n else float("nan")
    ci_high = mean + 1.96 * se if n else float("nan")
    z = mean / se if se > 0 else float("inf")
    p = two_sided_p_from_z(z) if math.isfinite(z) else 0.0
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "se": se,
        "ci95": [ci_low, ci_high],
        "z_vs_zero": z,
        "p_two_sided": p,
    }


def paired_stats(diffs: List[float]) -> Dict[str, float]:
    n = len(diffs)
    mean = statistics.fmean(diffs) if n else float("nan")
    sd = statistics.stdev(diffs) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    ci_low = mean - 1.96 * se if n else float("nan")
    ci_high = mean + 1.96 * se if n else float("nan")
    z = mean / se if se > 0 else float("inf")
    p = two_sided_p_from_z(z) if math.isfinite(z) else 0.0
    return {
        "n": n,
        "mean_diff": mean,
        "sd_diff": sd,
        "se_diff": se,
        "ci95": [ci_low, ci_high],
        "z": z,
        "p_two_sided": p,
    }


def build_summary(
    raw_rows: List[Dict[str, Any]],
    algorithms: List[str],
    seeds: List[int],
) -> Dict[str, Any]:
    by_algo_profile: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    by_seed_algo_profile: Dict[Tuple[int, str, str], Dict[str, Any]] = {}

    for row in raw_rows:
        key = (row["algorithm"], row["profile"])
        by_algo_profile[key].append(row)
        by_seed_algo_profile[(row["seed"], row["algorithm"], row["profile"])] = row

    per_algorithm_profile: Dict[str, Any] = defaultdict(dict)
    overall_by_algorithm: Dict[str, Any] = {}

    for algo in algorithms:
        per_seed_overall: List[float] = []
        for seed in seeds:
            seed_profile_vals: List[float] = []
            for profile in DEFAULT_PROFILES.keys():
                r = by_seed_algo_profile[(seed, algo, profile)]
                seed_profile_vals.append(float(r["bb_per_100"]))
            per_seed_overall.append(float(statistics.fmean(seed_profile_vals)))

        overall_by_algorithm[algo] = stats_against_zero(per_seed_overall)

        for profile in DEFAULT_PROFILES.keys():
            rows = by_algo_profile[(algo, profile)]
            bb = [float(r["bb_per_100"]) for r in rows]
            win = [float(r["win_rate"]) for r in rows]
            show = [float(r["showdown_rate"]) for r in rows]
            fold = [float(r.get("fold_rate", 0.0)) for r in rows]
            call = [float(r.get("call_rate", 0.0)) for r in rows]
            agg = stats_against_zero(bb)
            agg.update(
                {
                    "mean_win_rate": statistics.fmean(win),
                    "mean_showdown_rate": statistics.fmean(show),
                    "mean_fold_rate": statistics.fmean(fold),
                    "mean_call_rate": statistics.fmean(call),
                }
            )
            per_algorithm_profile[algo][profile] = agg

    pairwise_by_profile: Dict[str, Any] = defaultdict(dict)
    pairwise_overall: Dict[str, Any] = {}

    for a1, a2 in itertools.combinations(algorithms, 2):
        diff_overall: List[float] = []
        for seed in seeds:
            v1 = []
            v2 = []
            for profile in DEFAULT_PROFILES.keys():
                v1.append(float(by_seed_algo_profile[(seed, a1, profile)]["bb_per_100"]))
                v2.append(float(by_seed_algo_profile[(seed, a2, profile)]["bb_per_100"]))
            diff_overall.append(float(statistics.fmean(v1) - statistics.fmean(v2)))

        pairwise_overall[f"{a1}_minus_{a2}"] = paired_stats(diff_overall)

        for profile in DEFAULT_PROFILES.keys():
            diffs = []
            for seed in seeds:
                r1 = by_seed_algo_profile[(seed, a1, profile)]
                r2 = by_seed_algo_profile[(seed, a2, profile)]
                diffs.append(float(r1["bb_per_100"]) - float(r2["bb_per_100"]))
            pairwise_by_profile[profile][f"{a1}_minus_{a2}"] = paired_stats(diffs)

    return {
        "per_algorithm_profile": per_algorithm_profile,
        "overall_by_algorithm": overall_by_algorithm,
        "pairwise_by_profile": pairwise_by_profile,
        "pairwise_overall": pairwise_overall,
    }


def print_console_summary(summary: Dict[str, Any], algorithms: List[str]) -> None:
    print("\nOverall Mean bb/100 (across profiles, per-seed average)")
    for algo in algorithms:
        stats = summary["overall_by_algorithm"][algo]
        ci = stats["ci95"]
        print(
            f"- {algo:4s}: {stats['mean']:.1f} bb/100 "
            f"(95% CI [{ci[0]:.1f}, {ci[1]:.1f}], p={stats['p_two_sided']:.3g})"
        )

    print("\nPer-profile Mean bb/100")
    header = "profile".ljust(16) + " | " + " | ".join(a.ljust(8) for a in algorithms)
    print(header)
    print("-" * len(header))
    for profile in DEFAULT_PROFILES.keys():
        vals = []
        for algo in algorithms:
            vals.append(f"{summary['per_algorithm_profile'][algo][profile]['mean']:.1f}".ljust(8))
        print(profile.ljust(16) + " | " + " | ".join(vals))

    print("\nPaired Overall Comparisons (bb/100 diff)")
    for name, stats in summary["pairwise_overall"].items():
        ci = stats["ci95"]
        print(
            f"- {name}: {stats['mean_diff']:.1f} "
            f"(95% CI [{ci[0]:.1f}, {ci[1]:.1f}], p={stats['p_two_sided']:.3g})"
        )


def main() -> None:
    args = parse_args()
    algorithms = [a.strip().upper() for a in args.algorithms.split(",") if a.strip()]
    valid = {"PPO", "A2C", "DQN"}
    invalid = [a for a in algorithms if a not in valid]
    if invalid:
        raise ValueError(f"Unsupported algorithms: {invalid}. Allowed: {sorted(valid)}")

    seeds = list(range(args.seed_start, args.seed_start + max(1, args.seeds)))
    raw_rows: List[Dict[str, Any]] = []

    total_runs = len(algorithms) * len(seeds)
    run_idx = 0

    for algorithm in algorithms:
        for seed in seeds:
            run_idx += 1
            if not args.quiet:
                print(
                    f"[{run_idx}/{total_runs}] Running {algorithm} seed={seed} "
                    f"train_steps={args.train_steps} eval_hands={args.eval_hands}"
                )

            payload = run_experiment(
                algorithm=algorithm,
                train_profile=args.train_profile,
                train_steps=args.train_steps,
                eval_hands=args.eval_hands,
                seed=seed,
                device=args.device,
                print_output=False,
            )

            for row in payload["results"]:
                raw_rows.append(
                    {
                        "algorithm": algorithm,
                        "seed": seed,
                        **row,
                    }
                )

    summary = build_summary(raw_rows=raw_rows, algorithms=algorithms, seeds=seeds)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "algorithms": algorithms,
            "seed_start": args.seed_start,
            "n_seeds": len(seeds),
            "train_profile": args.train_profile,
            "train_steps": args.train_steps,
            "eval_hands": args.eval_hands,
            "device": args.device,
        },
        "summary": summary,
        "raw": raw_rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print_console_summary(summary, algorithms)
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Statistical evaluation harness for adaptive_chess.py."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import chess

import adaptive_chess as ac


DEFAULT_OUTPUT_PATH = Path(
    "/Users/sankalp/Projects/experiment/games/chess/eval/latest_eval.json"
)

TACTICAL_BENCHMARK_MAX_PLIES = 5
TACTICAL_BENCHMARK_TELL_SEED_OFFSET = 20_000
TACTICAL_BENCHMARK_FENS = [
    "8/5R2/1k6/1p2Q3/2K5/8/8/8 w - - 0 1",
    "1Q6/8/k4n2/4Q2K/1P6/8/8/8 w - - 0 1",
    "8/6rB/8/8/7K/8/2k5/6q1 b - - 0 1",
    "8/5p2/8/4Q1K1/8/1Q2q3/3k4/8 w - - 0 1",
    "5R2/8/5K1k/8/8/8/8/8 w - - 0 1",
    "8/2Q5/8/8/3kp1Q1/5K2/8/8 w - - 0 1",
]


@dataclass
class SeedEval:
    seed: int
    top1_correct: int
    top3_correct: int
    opponent_events: int
    nll_sum: float
    brier_sum: float
    mate_line_found: int
    white_turns: int
    final_result: str
    baseline_top1_expected: float
    baseline_top1_variance: float
    baseline_top3_expected: float
    baseline_top3_variance: float
    tactical_positions: int
    tactical_mate_line_found: int
    tactical_line_length_sum: int

    def top1_accuracy(self) -> Optional[float]:
        if self.opponent_events <= 0:
            return None
        return self.top1_correct / self.opponent_events

    def top3_accuracy(self) -> Optional[float]:
        if self.opponent_events <= 0:
            return None
        return self.top3_correct / self.opponent_events

    def nll(self) -> Optional[float]:
        if self.opponent_events <= 0:
            return None
        return self.nll_sum / self.opponent_events

    def brier(self) -> Optional[float]:
        if self.opponent_events <= 0:
            return None
        return self.brier_sum / self.opponent_events

    def mate_line_found_rate(self) -> Optional[float]:
        if self.white_turns <= 0:
            return None
        return self.mate_line_found / self.white_turns

    def as_dict(self) -> Dict[str, object]:
        return {
            "seed": self.seed,
            "top1_accuracy": self.top1_accuracy(),
            "top3_accuracy": self.top3_accuracy(),
            "nll": self.nll(),
            "brier": self.brier(),
            "mate_line_found_rate": self.mate_line_found_rate(),
            "counts": {
                "opponent_events": self.opponent_events,
                "white_turns": self.white_turns,
                "top1_correct": self.top1_correct,
                "top3_correct": self.top3_correct,
                "mate_line_found": self.mate_line_found,
            },
            "final_result": self.final_result,
        }


def _stats(values: List[float]) -> Dict[str, Optional[float]]:
    n = len(values)
    if n == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "ci95_low": None,
            "ci95_high": None,
        }

    mean = sum(values) / n
    if n == 1:
        std = 0.0
    else:
        var = sum((x - mean) ** 2 for x in values) / (n - 1)
        std = math.sqrt(max(0.0, var))
    half = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def _fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _significance_vs_random_legal_move(
    *,
    events: int,
    model_successes: int,
    baseline_expected_successes: float,
    baseline_variance: float,
) -> Dict[str, object]:
    z_score: Optional[float]
    p_value: Optional[float]
    if baseline_variance <= 0.0:
        z_score = None
        p_value = None
    else:
        z_score = (model_successes - baseline_expected_successes) / math.sqrt(baseline_variance)
        p_value = 0.5 * math.erfc(z_score / math.sqrt(2.0))
    return {
        "events": events,
        "model_successes": model_successes,
        "model_rate": (model_successes / events) if events > 0 else None,
        "baseline_expected_successes": baseline_expected_successes,
        "baseline_rate_expected": (baseline_expected_successes / events) if events > 0 else None,
        "z_score_normal_approx": z_score,
        "p_value_one_sided": p_value,
    }


def _run_tactical_mate_detection_benchmark(
    *,
    model: ac.AdaptiveMoveModel,
    profile: ac.PlayerProfile,
    seed: int,
    max_plies: int = TACTICAL_BENCHMARK_MAX_PLIES,
) -> Dict[str, int]:
    feed = ac.MockVideoFeed(
        profile=profile,
        seed=seed + TACTICAL_BENCHMARK_TELL_SEED_OFFSET,
        interval_s=10,
    )
    found = 0
    line_length_sum = 0
    for idx, fen in enumerate(TACTICAL_BENCHMARK_FENS):
        board = chess.Board(fen)
        tell = feed.snapshot_at(float(idx) * 10.0)
        line = ac.shortest_mate_line_under_expected_opponent(
            board,
            model,
            profile,
            us_color=board.turn,
            max_plies=max(2, max_plies),
            tell=tell,
        )
        if line:
            found += 1
            line_length_sum += len(line)
    return {
        "positions": len(TACTICAL_BENCHMARK_FENS),
        "found": found,
        "line_length_sum": line_length_sum,
    }


def evaluate_seed(
    *,
    profile: ac.PlayerProfile,
    seed: int,
    train_games: int,
    max_train_plies: int,
    max_demo_plies: int,
    max_mate_plies: int,
) -> SeedEval:
    model = ac.AdaptiveMoveModel()
    ac.train_adaptive_model(
        model=model,
        profile=profile,
        games=max(1, train_games),
        max_plies=max(10, max_train_plies),
        seed=seed,
    )
    tactical_benchmark = _run_tactical_mate_detection_benchmark(
        model=model,
        profile=profile,
        seed=seed,
    )

    rng = random.Random(seed + 999)
    true_policy = ac.TrueOpponentPolicy()
    feed = ac.MockVideoFeed(profile=profile, seed=seed + 555, interval_s=10)

    board = chess.Board()
    us_color = chess.WHITE

    top1_correct = 0
    top3_correct = 0
    opponent_events = 0
    nll_sum = 0.0
    brier_sum = 0.0
    mate_line_found = 0
    white_turns = 0
    baseline_top1_expected = 0.0
    baseline_top1_variance = 0.0
    baseline_top3_expected = 0.0
    baseline_top3_variance = 0.0

    ply = 0
    while not board.is_game_over() and ply < max(10, max_demo_plies):
        tell = feed.snapshot_at(ply * 10.0)

        if board.turn == us_color:
            white_turns += 1
            line = ac.shortest_mate_line_under_expected_opponent(
                board,
                model,
                profile,
                us_color=us_color,
                max_plies=max(2, max_mate_plies),
                tell=tell,
            )
            if line:
                mate_line_found += 1

            move = ac.pick_agent_move(
                board,
                model,
                profile,
                tell,
                us_color=us_color,
                max_mate_plies=max(2, max_mate_plies),
                rng=rng,
            )
            board.push(move)
        else:
            dist = model.predict_distribution(board, profile, tell)
            if not dist:
                break

            ranked = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
            top1_move = ranked[0][0]
            top3_moves = {mv for mv, _ in ranked[:3]}

            actual_move = true_policy.choose_move(board, profile, tell, rng)
            p_actual = max(1e-12, dist.get(actual_move, 0.0))
            sum_sq = sum(p * p for p in dist.values())

            opponent_events += 1
            top1_correct += int(actual_move == top1_move)
            top3_correct += int(actual_move in top3_moves)
            nll_sum += -math.log(p_actual)
            brier_sum += 1.0 - 2.0 * p_actual + sum_sq

            legal_count = max(1, len(dist))
            p_top1 = 1.0 / legal_count
            p_top3 = min(3, legal_count) / legal_count
            baseline_top1_expected += p_top1
            baseline_top1_variance += p_top1 * (1.0 - p_top1)
            baseline_top3_expected += p_top3
            baseline_top3_variance += p_top3 * (1.0 - p_top3)

            model.update(board, actual_move, tell)
            board.push(actual_move)

        ply += 1

    return SeedEval(
        seed=seed,
        top1_correct=top1_correct,
        top3_correct=top3_correct,
        opponent_events=opponent_events,
        nll_sum=nll_sum,
        brier_sum=brier_sum,
        mate_line_found=mate_line_found,
        white_turns=white_turns,
        final_result=board.result(claim_draw=True),
        baseline_top1_expected=baseline_top1_expected,
        baseline_top1_variance=baseline_top1_variance,
        baseline_top3_expected=baseline_top3_expected,
        baseline_top3_variance=baseline_top3_variance,
        tactical_positions=tactical_benchmark["positions"],
        tactical_mate_line_found=tactical_benchmark["found"],
        tactical_line_length_sum=tactical_benchmark["line_length_sum"],
    )


def _build_report(
    *,
    profile_name: str,
    seed_results: List[SeedEval],
    train_games: int,
    max_train_plies: int,
    max_demo_plies: int,
    max_mate_plies: int,
) -> Dict[str, object]:
    top1_values = [x for x in (r.top1_accuracy() for r in seed_results) if x is not None]
    top3_values = [x for x in (r.top3_accuracy() for r in seed_results) if x is not None]
    nll_values = [x for x in (r.nll() for r in seed_results) if x is not None]
    brier_values = [x for x in (r.brier() for r in seed_results) if x is not None]
    mate_values = [x for x in (r.mate_line_found_rate() for r in seed_results) if x is not None]
    tactical_rate_values = [
        (r.tactical_mate_line_found / r.tactical_positions)
        for r in seed_results
        if r.tactical_positions > 0
    ]
    tactical_avg_len_values = [
        (r.tactical_line_length_sum / r.tactical_mate_line_found)
        for r in seed_results
        if r.tactical_mate_line_found > 0
    ]

    total_events = sum(r.opponent_events for r in seed_results)
    total_top1 = sum(r.top1_correct for r in seed_results)
    total_top3 = sum(r.top3_correct for r in seed_results)
    baseline_top1_expected = sum(r.baseline_top1_expected for r in seed_results)
    baseline_top1_variance = sum(r.baseline_top1_variance for r in seed_results)
    baseline_top3_expected = sum(r.baseline_top3_expected for r in seed_results)
    baseline_top3_variance = sum(r.baseline_top3_variance for r in seed_results)

    total_tactical_positions = sum(r.tactical_positions for r in seed_results)
    total_tactical_found = sum(r.tactical_mate_line_found for r in seed_results)
    total_tactical_line_length_sum = sum(r.tactical_line_length_sum for r in seed_results)

    result_counts = Counter(r.final_result for r in seed_results)
    result_rates = {
        k: (v / len(seed_results) if seed_results else None)
        for k, v in sorted(result_counts.items(), key=lambda kv: kv[0])
    }

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "profile": profile_name,
            "seeds": len(seed_results),
            "train_games": train_games,
            "max_train_plies": max_train_plies,
            "max_demo_plies": max_demo_plies,
            "max_mate_plies": max_mate_plies,
            "run_mode": "in_process_non_verbose_import",
        },
        "aggregate": {
            "top1_accuracy": _stats(top1_values),
            "top3_accuracy": _stats(top3_values),
            "nll": _stats(nll_values),
            "brier": _stats(brier_values),
            "mate_line_found_rate": _stats(mate_values),
            "random_legal_move_baseline_rate_expected": {
                "top1": (baseline_top1_expected / total_events) if total_events > 0 else None,
                "top3": (baseline_top3_expected / total_events) if total_events > 0 else None,
            },
            "mate_line_found_rate_on_tactical_positions": _stats(tactical_rate_values),
            "avg_line_length_when_found_on_tactical_positions": _stats(tactical_avg_len_values),
            "final_result_distribution": {
                "counts": dict(result_counts),
                "rates": result_rates,
            },
        },
        "significance": {
            "top1_vs_random_legal_move": _significance_vs_random_legal_move(
                events=total_events,
                model_successes=total_top1,
                baseline_expected_successes=baseline_top1_expected,
                baseline_variance=baseline_top1_variance,
            ),
            "top3_vs_random_legal_move": _significance_vs_random_legal_move(
                events=total_events,
                model_successes=total_top3,
                baseline_expected_successes=baseline_top3_expected,
                baseline_variance=baseline_top3_variance,
            ),
        },
        "benchmarks": {
            "tactical_mate_detection": {
                "positions_total": total_tactical_positions,
                "positions_per_seed": len(TACTICAL_BENCHMARK_FENS),
                "max_plies": TACTICAL_BENCHMARK_MAX_PLIES,
                "mate_line_found_count": total_tactical_found,
                "mate_line_found_rate_on_tactical_positions": (
                    total_tactical_found / total_tactical_positions
                )
                if total_tactical_positions > 0
                else None,
                "avg_line_length_when_found": (
                    total_tactical_line_length_sum / total_tactical_found
                )
                if total_tactical_found > 0
                else None,
            }
        },
        "per_seed": [r.as_dict() for r in seed_results],
    }


def _print_summary(report: Mapping[str, object], output_path: Path) -> None:
    config = report["config"]  # type: ignore[index]
    aggregate = report["aggregate"]  # type: ignore[index]
    significance = report["significance"]  # type: ignore[index]

    print(
        "Profile={profile}  Seeds={seeds}  TrainGames={train_games}  DemoPlies={demo}".format(
            profile=config["profile"],  # type: ignore[index]
            seeds=config["seeds"],  # type: ignore[index]
            train_games=config["train_games"],  # type: ignore[index]
            demo=config["max_demo_plies"],  # type: ignore[index]
        )
    )
    print("Metric                 mean      std       95% CI")
    print("---------------------------------------------------------")

    for key in ("top1_accuracy", "top3_accuracy", "nll", "brier", "mate_line_found_rate"):
        row = aggregate[key]  # type: ignore[index]
        print(
            f"{key:<21} "
            f"{_fmt(row['mean']):>8}  "
            f"{_fmt(row['std']):>8}  "
            f"[{_fmt(row['ci95_low'])}, {_fmt(row['ci95_high'])}]"
        )

    sig_top1 = significance["top1_vs_random_legal_move"]  # type: ignore[index]
    sig_top3 = significance["top3_vs_random_legal_move"]  # type: ignore[index]
    print("---------------------------------------------------------")
    print(
        "top1 > random-legal baseline: "
        f"model={_fmt(sig_top1['model_rate'])}, "
        f"baseline={_fmt(sig_top1['baseline_rate_expected'])}, "
        f"z={_fmt(sig_top1['z_score_normal_approx'], digits=3)}, "
        f"p={_fmt(sig_top1['p_value_one_sided'], digits=6)}"
    )
    print(
        "top3 > random-legal baseline: "
        f"model={_fmt(sig_top3['model_rate'])}, "
        f"baseline={_fmt(sig_top3['baseline_rate_expected'])}, "
        f"z={_fmt(sig_top3['z_score_normal_approx'], digits=3)}, "
        f"p={_fmt(sig_top3['p_value_one_sided'], digits=6)}"
    )

    dist = aggregate["final_result_distribution"]  # type: ignore[index]
    counts = dist["counts"]  # type: ignore[index]
    if counts:
        ordered = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items(), key=lambda kv: kv[0]))
        print(f"final_result_distribution: {ordered}")
    else:
        print("final_result_distribution: n/a")

    benchmarks = report["benchmarks"]  # type: ignore[index]
    tactical = benchmarks["tactical_mate_detection"]  # type: ignore[index]
    print(
        "tactical_mate_detection: "
        f"found_rate={_fmt(tactical['mate_line_found_rate_on_tactical_positions'])}, "
        f"avg_len={_fmt(tactical['avg_line_length_when_found'])}, "
        f"found={tactical['mate_line_found_count']}/{tactical['positions_total']}"
    )

    print(f"report_json: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Statistical evaluator for adaptive_chess.py")
    parser.add_argument(
        "--profile",
        type=str,
        default="balanced",
        choices=sorted(ac.DEFAULT_PROFILES.keys()),
    )
    parser.add_argument("--seeds", type=int, default=20, help="Number of seeds to evaluate.")
    parser.add_argument(
        "--seed-start",
        type=int,
        default=42,
        help="First seed value. Seeds are [seed_start, seed_start + seeds).",
    )
    parser.add_argument("--train-games", type=int, default=300)
    parser.add_argument("--max-train-plies", type=int, default=80)
    parser.add_argument(
        "--eval-plies",
        "--max-demo-plies",
        dest="max_demo_plies",
        type=int,
        default=40,
        help="Maximum demo plies to evaluate per seed.",
    )
    parser.add_argument("--max-mate-plies", type=int, default=6)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = ac.DEFAULT_PROFILES[args.profile]

    seed_results: List[SeedEval] = []
    for seed in range(args.seed_start, args.seed_start + max(1, args.seeds)):
        result = evaluate_seed(
            profile=profile,
            seed=seed,
            train_games=args.train_games,
            max_train_plies=args.max_train_plies,
            max_demo_plies=args.max_demo_plies,
            max_mate_plies=args.max_mate_plies,
        )
        seed_results.append(result)

    report = _build_report(
        profile_name=profile.name,
        seed_results=seed_results,
        train_games=args.train_games,
        max_train_plies=args.max_train_plies,
        max_demo_plies=args.max_demo_plies,
        max_mate_plies=args.max_mate_plies,
    )

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    _print_summary(report, output_path)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, List

from shared.game_specs import get_missed_game_names, list_game_names


def _parse_games(csv_text: str | None) -> List[str]:
    if csv_text is None or not csv_text.strip():
        return get_missed_game_names()

    requested = [part.strip().lower() for part in csv_text.split(",") if part.strip()]
    available = set(list_game_names())
    unknown = [name for name in requested if name not in available]
    if unknown:
        raise ValueError(f"Unknown games: {unknown}. Available: {', '.join(sorted(available))}")
    return requested


def _run_single_game(
    game: str,
    python_executable: str,
    output_root: Path,
    seeds: int,
    seed_start: int,
    train_steps: int,
    eval_steps: int,
    algorithms: str,
    tell_scales: str,
    cwd: Path,
) -> Dict[str, object]:
    run_dir = output_root / game
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        python_executable,
        "-m",
        "shared.train_multi_method",
        "--game",
        game,
        "--seeds",
        str(seeds),
        "--seed-start",
        str(seed_start),
        "--train-steps",
        str(train_steps),
        "--eval-steps",
        str(eval_steps),
        "--algorithms",
        algorithms,
        "--tell-scales",
        tell_scales,
        "--output-root",
        str(output_root),
        "--quiet",
    ]

    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    elapsed = time.perf_counter() - started

    log_path = run_dir / "orchestrator_subprocess.log"
    log_path.write_text(
        "\n".join(
            [
                f"command: {' '.join(cmd)}",
                f"returncode: {proc.returncode}",
                "",
                "[stdout]",
                proc.stdout,
                "",
                "[stderr]",
                proc.stderr,
            ]
        ),
        encoding="utf-8",
    )

    metrics_path = run_dir / "metrics.json"
    result: Dict[str, object] = {
        "game": game,
        "command": cmd,
        "returncode": int(proc.returncode),
        "elapsed_seconds": float(elapsed),
        "metrics_path": str(metrics_path),
        "log_path": str(log_path),
        "status": "ok" if proc.returncode == 0 else "failed",
    }

    if proc.returncode == 0 and metrics_path.exists():
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        per_algorithm = data.get("per_algorithm", {})
        best_algorithm = None
        best_mean = float("-inf")

        for algorithm, stats in per_algorithm.items():
            mean = float(stats.get("mean", float("-inf")))
            if mean > best_mean:
                best_mean = mean
                best_algorithm = str(algorithm)

        result["best_algorithm"] = best_algorithm
        result["best_mean_reward"] = best_mean

    return result


def _write_summary_markdown(summary: Dict[str, object], output_path: Path) -> None:
    rows = summary["games"]

    lines = [
        "# Multi-Game Summary",
        "",
        f"Generated: {summary['generated_at_utc']}",
        "",
        "| Game | Status | Best Algorithm | Best Mean Reward | Elapsed (s) |",
        "|---|---|---|---:|---:|",
    ]

    for row in rows:
        game = row.get("game", "")
        status = row.get("status", "")
        best_algorithm = row.get("best_algorithm", "-")
        best_mean = row.get("best_mean_reward", float("nan"))
        elapsed = row.get("elapsed_seconds", 0.0)
        if isinstance(best_mean, (int, float)):
            best_mean_text = f"{best_mean:+.4f}"
        else:
            best_mean_text = "-"
        lines.append(
            f"| {game} | {status} | {best_algorithm} | {best_mean_text} | {float(elapsed):.2f} |"
        )

    if summary.get("failures"):
        lines.extend(["", "## Failures", ""])
        for failure in summary["failures"]:
            lines.append(f"- {failure['game']} (see `{failure['log_path']}`)")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run all missed games using shared multi-method harness")
    parser.add_argument("--games", type=str, default=None, help="Comma-separated game names. Defaults to all missed games.")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--train-steps", type=int, default=280)
    parser.add_argument("--eval-steps", type=int, default=140)
    parser.add_argument("--algorithms", type=str, default="PPO,A2C,DQN")
    parser.add_argument("--tell-scales", type=str, default="0.0,0.5,1.0,1.5")
    parser.add_argument("--jobs", type=int, default=max(1, min(4, (os.cpu_count() or 1))))
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    games = _parse_games(args.games)
    if not games:
        raise ValueError("No games selected")

    jobs = max(1, min(int(args.jobs), len(games)))
    cwd = Path(__file__).resolve().parents[1]

    print(f"running games={','.join(games)} jobs={jobs}")

    results: List[Dict[str, object]] = []
    started = time.perf_counter()

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        future_map = {
            executor.submit(
                _run_single_game,
                game,
                sys.executable,
                output_root,
                max(1, int(args.seeds)),
                int(args.seed_start),
                max(1, int(args.train_steps)),
                max(1, int(args.eval_steps)),
                args.algorithms,
                args.tell_scales,
                cwd,
            ): game
            for game in games
        }

        for future in as_completed(future_map):
            result = future.result()
            results.append(result)
            print(
                f"[{result['status']}] {result['game']} "
                f"elapsed={result['elapsed_seconds']:.2f}s returncode={result['returncode']}"
            )

    total_elapsed = time.perf_counter() - started
    results.sort(key=lambda row: str(row["game"]))

    failures = [row for row in results if row["status"] != "ok"]

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "games": games,
            "seeds": max(1, int(args.seeds)),
            "seed_start": int(args.seed_start),
            "train_steps": max(1, int(args.train_steps)),
            "eval_steps": max(1, int(args.eval_steps)),
            "algorithms": [part.strip().upper() for part in args.algorithms.split(",") if part.strip()],
            "tell_scales": [float(part.strip()) for part in args.tell_scales.split(",") if part.strip()],
            "jobs": jobs,
            "output_root": str(output_root),
        },
        "elapsed_seconds": float(total_elapsed),
        "games": results,
        "failures": failures,
    }

    json_path = output_root / "summary_report.json"
    md_path = output_root / "summary_report.md"

    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_summary_markdown(summary=summary, output_path=md_path)

    print(f"summary_json={json_path}")
    print(f"summary_md={md_path}")
    print(f"total_elapsed_s={total_elapsed:.2f}")


if __name__ == "__main__":
    main()

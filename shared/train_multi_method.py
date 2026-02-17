from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import List

from shared.eval_harness import run_game_protocol, save_metrics
from shared.game_specs import get_game_spec, list_game_names
from shared.methods import ALGORITHM_NAMES
from shared.visualize import generate_all_plots


def _parse_algorithms(text: str) -> List[str]:
    algorithms = [part.strip().upper() for part in text.split(",") if part.strip()]
    if not algorithms:
        raise ValueError("At least one algorithm must be provided")
    unknown = [name for name in algorithms if name not in ALGORITHM_NAMES]
    if unknown:
        raise ValueError(f"Unsupported algorithms: {unknown}. Allowed: {', '.join(ALGORITHM_NAMES)}")
    return algorithms


def _parse_float_csv(text: str) -> List[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one tell scale is required")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lightweight multi-method simulation harness")
    parser.add_argument("--game", required=True, choices=list_game_names())
    parser.add_argument("--algorithms", default="PPO,A2C,DQN")
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--train-steps", type=int, default=280)
    parser.add_argument("--eval-steps", type=int, default=140)
    parser.add_argument("--tell-scales", type=str, default="0.0,0.5,1.0,1.5")
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    algorithms = _parse_algorithms(args.algorithms)
    tell_scales = _parse_float_csv(args.tell_scales)

    n_seeds = max(1, int(args.seeds))
    seed_start = int(args.seed_start)
    seeds = list(range(seed_start, seed_start + n_seeds))

    spec = get_game_spec(args.game)
    run_dir = Path(args.output_root).expanduser().resolve() / spec.name
    metrics_path = run_dir / "metrics.json"

    started = time.perf_counter()
    metrics = run_game_protocol(
        game_name=spec.name,
        algorithms=algorithms,
        seeds=seeds,
        train_steps=max(1, int(args.train_steps)),
        eval_steps=max(1, int(args.eval_steps)),
        tell_scales=tell_scales,
    )

    save_metrics(metrics=metrics, output_path=metrics_path)
    plot_paths = generate_all_plots(metrics=metrics, run_dir=run_dir)
    elapsed = time.perf_counter() - started

    if not args.quiet:
        print(f"game={spec.name} seeds={len(seeds)} algorithms={','.join(algorithms)}")
        for algorithm in algorithms:
            stats = metrics["per_algorithm"][algorithm]
            ci = stats["ci95"]
            print(
                f"{algorithm:>4s} mean={stats['mean']:+.4f} sd={stats['sd']:.4f} se={stats['se']:.4f} "
                f"ci95=[{ci[0]:+.4f},{ci[1]:+.4f}] z={stats['z_vs_zero']:+.3f} p={stats['p_vs_zero']:.3g}"
            )

        print("pairwise:")
        for name, stats in metrics["pairwise"].items():
            ci = stats["ci95"]
            print(
                f"  {name}: mean_diff={stats['mean_diff']:+.4f} "
                f"ci95=[{ci[0]:+.4f},{ci[1]:+.4f}] z={stats['z_vs_zero']:+.3f} p={stats['p_vs_zero']:.3g}"
            )

        print(f"metrics={metrics_path}")
        print("plots=" + ",".join(str(path) for path in plot_paths))
        print(f"elapsed_s={elapsed:.2f}")


if __name__ == "__main__":
    main()

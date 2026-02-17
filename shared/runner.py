from __future__ import annotations

from pathlib import Path
from typing import Sequence

from shared.eval_harness import run_game_protocol, save_metrics
from shared.visualize import generate_all_plots


def _normalize_algorithms(algorithms: Sequence[str]) -> list[str]:
    return [name.strip().upper() for name in algorithms if name.strip()]


def run_game(
    *,
    game_name: str,
    seeds: int,
    algorithms: Sequence[str],
    train_episodes: int,
    eval_episodes: int,
    video: bool,
    output_dir: str | None,
) -> dict[str, object]:
    del video  # The mock-video feed is always part of the state schema in this harness.

    root = Path(__file__).resolve().parents[1]
    run_dir = Path(output_dir).expanduser().resolve() if output_dir else root / game_name / "outputs"
    run_dir.mkdir(parents=True, exist_ok=True)

    seed_count = max(1, int(seeds))
    seed_list = list(range(1, seed_count + 1))

    metrics = run_game_protocol(
        game_name=game_name,
        algorithms=_normalize_algorithms(algorithms),
        seeds=seed_list,
        train_steps=max(1, int(train_episodes)),
        eval_steps=max(1, int(eval_episodes)),
        tell_scales=[0.0, 0.5, 1.0, 1.5],
    )

    metrics_path = run_dir / "metrics.json"
    save_metrics(metrics=metrics, output_path=metrics_path)
    plot_paths = generate_all_plots(metrics=metrics, run_dir=run_dir)

    return {
        "metrics_path": str(metrics_path),
        "plots": [str(path) for path in plot_paths],
    }

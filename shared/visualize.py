from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _algorithm_order(metrics: Mapping[str, object]) -> List[str]:
    config = metrics.get("config", {})
    configured = config.get("algorithms", [])
    if isinstance(configured, list) and configured:
        return [str(name) for name in configured]
    per_algorithm = metrics.get("per_algorithm", {})
    return sorted(str(name) for name in per_algorithm)


def plot_reward_ci(metrics: Mapping[str, object], output_path: Path) -> None:
    algorithms = _algorithm_order(metrics)
    per_algorithm: Dict[str, Mapping[str, object]] = metrics.get("per_algorithm", {})  # type: ignore[assignment]

    means = [float(per_algorithm[algo]["mean"]) for algo in algorithms]
    cis = [per_algorithm[algo]["ci95"] for algo in algorithms]
    lower = [mean - float(ci[0]) for mean, ci in zip(means, cis)]
    upper = [float(ci[1]) - mean for mean, ci in zip(means, cis)]

    x = np.arange(len(algorithms), dtype=float)

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.bar(x, means, color=["#4C78A8", "#F58518", "#54A24B"][: len(algorithms)], alpha=0.85)
    ax.errorbar(x, means, yerr=[lower, upper], fmt="none", capsize=4, color="black", linewidth=1.1)
    ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--")

    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.set_ylabel("Mean reward")
    ax.set_title("Mean Reward with 95% CI")
    ax.grid(axis="y", alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_method_comparison(metrics: Mapping[str, object], output_path: Path) -> None:
    algorithms = _algorithm_order(metrics)
    seed_rows: Sequence[Mapping[str, object]] = metrics.get("seed_results", [])  # type: ignore[assignment]

    by_algorithm: Dict[str, List[float]] = defaultdict(list)
    for row in seed_rows:
        algorithm = str(row.get("algorithm"))
        if algorithm in algorithms:
            by_algorithm[algorithm].append(float(row.get("mean_reward", 0.0)))

    values = [by_algorithm[algorithm] for algorithm in algorithms]

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    box = ax.boxplot(values, patch_artist=True, labels=algorithms)
    palette = ["#4C78A8", "#F58518", "#54A24B"]
    for patch, color in zip(box["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.70)

    ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--")
    ax.set_ylabel("Per-seed mean reward")
    ax.set_title("Method Comparison Across Seeds")
    ax.grid(axis="y", alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_tell_sensitivity(metrics: Mapping[str, object], output_path: Path) -> None:
    algorithms = _algorithm_order(metrics)
    tell = metrics.get("tell_sensitivity", {})
    per_algorithm: Mapping[str, Sequence[Mapping[str, object]]] = tell.get("per_algorithm", {})  # type: ignore[assignment]

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)

    palette = {
        "PPO": "#4C78A8",
        "A2C": "#F58518",
        "DQN": "#54A24B",
    }

    for algorithm in algorithms:
        rows = per_algorithm.get(algorithm, [])
        scales = [float(row.get("tell_scale", 0.0)) for row in rows]
        means = [float(row.get("mean", 0.0)) for row in rows]
        ses = [float(row.get("se", 0.0)) for row in rows]

        color = palette.get(algorithm, "#333333")
        ax.plot(scales, means, marker="o", linewidth=1.8, color=color, label=algorithm)
        if scales:
            lower = [m - 1.96 * se for m, se in zip(means, ses)]
            upper = [m + 1.96 * se for m, se in zip(means, ses)]
            ax.fill_between(scales, lower, upper, color=color, alpha=0.14)

    ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Tell scale")
    ax.set_ylabel("Mean reward")
    ax.set_title("Tell Sensitivity")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def generate_all_plots(metrics: Mapping[str, object], run_dir: Path) -> List[Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        run_dir / "reward_ci.png",
        run_dir / "method_comparison.png",
        run_dir / "tell_sensitivity.png",
    ]
    plot_reward_ci(metrics=metrics, output_path=outputs[0])
    plot_method_comparison(metrics=metrics, output_path=outputs[1])
    plot_tell_sensitivity(metrics=metrics, output_path=outputs[2])
    return outputs

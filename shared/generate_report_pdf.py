"""Generate a comprehensive PDF report from all game run results.

Usage:
    python -m shared.generate_report_pdf --output-root runs
"""

from __future__ import annotations

import argparse
import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_game_metrics(runs_dir: Path) -> List[Dict[str, Any]]:
    """Load all metrics.json files from game sub-directories."""
    results: List[Dict[str, Any]] = []
    for child in sorted(runs_dir.iterdir()):
        metrics_path = child / "metrics.json"
        if child.is_dir() and metrics_path.exists():
            with open(metrics_path) as f:
                data = json.load(f)
            data["_dir"] = child
            results.append(data)
    return results


def _best_algo(metrics: Dict[str, Any]) -> Tuple[str, float]:
    """Return (algorithm_name, mean_reward) for the best algorithm."""
    per_algo = metrics.get("per_algorithm", {})
    best_name, best_mean = "", float("-inf")
    for name, stats in per_algo.items():
        m = float(stats["mean"])
        if m > best_mean:
            best_name, best_mean = name, m
    return best_name, best_mean


def _fmt(val: float, decimals: int = 4) -> str:
    return f"{val:+.{decimals}f}" if val != 0 else f"{val:.{decimals}f}"


def _fmt_p(p: float) -> str:
    if p == 0:
        return "< 1e-16"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def _sig_marker(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ---------------------------------------------------------------------------
# Page builders
# ---------------------------------------------------------------------------

def _page_title(pdf: PdfPages, all_metrics: List[Dict[str, Any]]) -> None:
    """Create a title page."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")

    # Gather config summary from first game
    config = all_metrics[0].get("config", {}) if all_metrics else {}
    algos = config.get("algorithms", [])
    n_seeds = config.get("n_seeds", "?")
    train_steps = config.get("train_steps", "?")
    eval_steps = config.get("eval_steps", "?")
    tell_scales = config.get("tell_scales", [])

    lines = [
        ("Multi-Method Game Simulation Report", 22, 0.82, "bold"),
        ("", 12, 0.78, "normal"),
        (f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", 11, 0.74, "normal"),
        (f"Games analyzed: {len(all_metrics)}", 11, 0.70, "normal"),
        (f"Algorithms: {', '.join(algos)}", 11, 0.66, "normal"),
        (f"Seeds per game: {n_seeds}   |   Train steps: {train_steps}   |   Eval steps: {eval_steps}", 11, 0.62, "normal"),
        (f"Tell scales: {tell_scales}", 11, 0.58, "normal"),
    ]

    for text, size, y, weight in lines:
        ax.text(0.5, y, text, ha="center", va="center", fontsize=size,
                fontweight=weight, transform=ax.transAxes)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _page_summary_table(pdf: PdfPages, all_metrics: List[Dict[str, Any]]) -> None:
    """Create a summary table of all games sorted by best mean reward."""
    ranked = []
    for m in all_metrics:
        game = m.get("game", "?")
        best_name, best_mean = _best_algo(m)
        per_algo = m.get("per_algorithm", {})
        ci = per_algo.get(best_name, {}).get("ci95", [None, None])
        p = per_algo.get(best_name, {}).get("p_vs_zero", 1.0)
        ranked.append((game, best_name, best_mean, ci, p))

    ranked.sort(key=lambda r: r[2], reverse=True)

    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    ax.set_title("Summary: All Games Ranked by Best Mean Reward", fontsize=14,
                 fontweight="bold", pad=20)

    col_labels = ["Rank", "Game", "Best Algo", "Mean", "95% CI", "p(vs 0)"]
    cell_text = []
    for i, (game, algo, mean, ci, p) in enumerate(ranked, 1):
        ci_str = f"[{_fmt(ci[0])}, {_fmt(ci[1])}]" if ci[0] is not None else "N/A"
        cell_text.append([
            str(i), game, algo, _fmt(mean), ci_str,
            _fmt_p(p) + _sig_marker(p),
        ])

    table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.35)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#4C78A8")
        cell.set_text_props(color="white", fontweight="bold")

    # Alternate row colors
    for i in range(1, len(cell_text) + 1):
        color = "#f0f4fa" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(color)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _page_game_header(pdf: PdfPages, metrics: Dict[str, Any]) -> None:
    """Header page for a game with stats table + all 3 charts."""
    game = metrics.get("game", "?")
    game_dir: Path = metrics["_dir"]
    per_algo = metrics.get("per_algorithm", {})
    pairwise = metrics.get("pairwise", {})
    spec = metrics.get("state_schema", {}).get("game_spec", {})
    algos = list(per_algo.keys())

    # --- Page 1: Header + stats tables ---
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 11),
                             gridspec_kw={"height_ratios": [1, 1]})

    # Top half: per-algorithm stats
    ax_top = axes[0]
    ax_top.axis("off")

    title_parts = [f"Game: {game}"]
    if spec:
        title_parts.append(
            f"difficulty={spec.get('difficulty', '?')}  "
            f"tell_weight={spec.get('tell_weight', '?')}  "
            f"profile_weight={spec.get('profile_weight', '?')}"
        )
    ax_top.set_title("\n".join(title_parts), fontsize=13, fontweight="bold",
                     loc="left", pad=10)

    col_labels = ["Algorithm", "Mean", "SD", "SE", "95% CI", "p(vs 0)", "n"]
    rows = []
    for algo in sorted(algos):
        s = per_algo[algo]
        ci = s.get("ci95", [None, None])
        ci_str = f"[{_fmt(ci[0])}, {_fmt(ci[1])}]" if ci[0] is not None else "N/A"
        rows.append([
            algo, _fmt(s["mean"]), f"{s['sd']:.4f}", f"{s['se']:.4f}",
            ci_str, _fmt_p(s["p_vs_zero"]) + _sig_marker(s["p_vs_zero"]),
            str(s["n"]),
        ])

    t1 = ax_top.table(cellText=rows, colLabels=col_labels, loc="center",
                      cellLoc="center")
    t1.auto_set_font_size(False)
    t1.set_fontsize(8)
    t1.scale(1.0, 1.5)
    for j in range(len(col_labels)):
        t1[0, j].set_facecolor("#4C78A8")
        t1[0, j].set_text_props(color="white", fontweight="bold")

    # Bottom half: pairwise comparisons
    ax_bot = axes[1]
    ax_bot.axis("off")
    ax_bot.set_title("Pairwise Comparisons", fontsize=11, fontweight="bold",
                     loc="left", pad=10)

    pw_labels = ["Comparison", "Mean Diff", "95% CI", "p(vs 0)"]
    pw_rows = []
    for pair_name in sorted(pairwise.keys()):
        pw = pairwise[pair_name]
        ci = pw.get("ci95", [None, None])
        ci_str = f"[{_fmt(ci[0])}, {_fmt(ci[1])}]" if ci[0] is not None else "N/A"
        pw_rows.append([
            pair_name.replace("_minus_", " - "),
            _fmt(pw["mean_diff"]),
            ci_str,
            _fmt_p(pw["p_vs_zero"]) + _sig_marker(pw["p_vs_zero"]),
        ])

    if pw_rows:
        t2 = ax_bot.table(cellText=pw_rows, colLabels=pw_labels, loc="center",
                          cellLoc="center")
        t2.auto_set_font_size(False)
        t2.set_fontsize(8)
        t2.scale(1.0, 1.5)
        for j in range(len(pw_labels)):
            t2[0, j].set_facecolor("#F58518")
            t2[0, j].set_text_props(color="white", fontweight="bold")
    else:
        ax_bot.text(0.5, 0.5, "No pairwise comparisons available.",
                    ha="center", va="center", fontsize=10)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # --- Page 2: Charts ---
    chart_files = ["reward_ci.png", "method_comparison.png", "tell_sensitivity.png"]
    chart_titles = ["Mean Reward with 95% CI", "Method Comparison Across Seeds",
                    "Tell Sensitivity"]

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 11))
    fig.suptitle(f"{game} - Charts", fontsize=14, fontweight="bold", y=0.98)

    for ax, fname, ctitle in zip(axes, chart_files, chart_titles):
        img_path = game_dir / fname
        if img_path.exists():
            img = plt.imread(str(img_path))
            ax.imshow(img, aspect="auto")
            ax.set_title(ctitle, fontsize=9, pad=3)
        else:
            ax.text(0.5, 0.5, f"{fname} not found", ha="center", va="center",
                    fontsize=10)
        ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    pdf.savefig(fig)
    plt.close(fig)


def _page_cross_game(pdf: PdfPages, all_metrics: List[Dict[str, Any]]) -> None:
    """Cross-game comparison: win counts, mean reward distribution."""
    # Count wins per algorithm
    win_counts: Dict[str, int] = {}
    algo_means: Dict[str, List[float]] = {}

    for m in all_metrics:
        per_algo = m.get("per_algorithm", {})
        for algo, stats in per_algo.items():
            algo_means.setdefault(algo, []).append(float(stats["mean"]))
        best_name, _ = _best_algo(m)
        win_counts[best_name] = win_counts.get(best_name, 0) + 1

    algos_sorted = sorted(win_counts.keys())

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 11))
    fig.suptitle("Cross-Game Comparison", fontsize=14, fontweight="bold", y=0.98)

    # --- Win count bar chart ---
    ax = axes[0, 0]
    palette = {"PPO": "#4C78A8", "A2C": "#F58518", "DQN": "#54A24B"}
    names = sorted(win_counts.keys())
    counts = [win_counts[n] for n in names]
    colors = [palette.get(n, "#888888") for n in names]
    ax.bar(names, counts, color=colors, alpha=0.85)
    ax.set_title("Games Won per Algorithm", fontsize=10, fontweight="bold")
    ax.set_ylabel("# games with highest mean")
    ax.grid(axis="y", alpha=0.25)
    for i, (n, c) in enumerate(zip(names, counts)):
        ax.text(i, c + 0.15, str(c), ha="center", fontsize=10, fontweight="bold")

    # --- Mean reward distribution (box plots per algorithm) ---
    ax = axes[0, 1]
    box_data = [algo_means.get(a, []) for a in algos_sorted]
    box = ax.boxplot(box_data, patch_artist=True, tick_labels=algos_sorted)
    for patch, algo in zip(box["boxes"], algos_sorted):
        patch.set_facecolor(palette.get(algo, "#888888"))
        patch.set_alpha(0.7)
    ax.axhline(0, color="#666", ls="--", lw=0.8)
    ax.set_title("Mean Reward Distribution Across Games", fontsize=10,
                 fontweight="bold")
    ax.set_ylabel("Mean reward")
    ax.grid(axis="y", alpha=0.25)

    # --- Average mean reward per algorithm ---
    ax = axes[1, 0]
    avg_means = {a: float(np.mean(vals)) for a, vals in algo_means.items()}
    avg_se = {a: float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
              for a, vals in algo_means.items() if len(vals) > 1}
    names_avg = sorted(avg_means.keys())
    avgs = [avg_means[n] for n in names_avg]
    ses = [avg_se.get(n, 0) for n in names_avg]
    colors = [palette.get(n, "#888888") for n in names_avg]
    ax.bar(names_avg, avgs, yerr=ses, color=colors, alpha=0.85, capsize=4)
    ax.axhline(0, color="#666", ls="--", lw=0.8)
    ax.set_title("Grand Mean Reward per Algorithm", fontsize=10,
                 fontweight="bold")
    ax.set_ylabel("Mean reward (avg across games)")
    ax.grid(axis="y", alpha=0.25)

    # --- Positive vs negative games per algorithm ---
    ax = axes[1, 1]
    pos_counts = {a: sum(1 for v in vals if v > 0) for a, vals in algo_means.items()}
    neg_counts = {a: sum(1 for v in vals if v <= 0) for a, vals in algo_means.items()}
    x = np.arange(len(algos_sorted))
    width = 0.35
    ax.bar(x - width / 2, [pos_counts.get(a, 0) for a in algos_sorted],
           width, label="Positive mean", color="#54A24B", alpha=0.8)
    ax.bar(x + width / 2, [neg_counts.get(a, 0) for a in algos_sorted],
           width, label="Negative/zero mean", color="#E45756", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(algos_sorted)
    ax.set_title("Games with Positive vs Negative Mean Reward", fontsize=10,
                 fontweight="bold")
    ax.set_ylabel("# games")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig)
    plt.close(fig)


def _page_gap_analysis(pdf: PdfPages, runs_dir: Path) -> None:
    """Render gap_analysis.md content onto PDF pages."""
    gap_path = runs_dir / "gap_analysis.md"
    if not gap_path.exists():
        return

    text = gap_path.read_text(encoding="utf-8")
    if not text.strip():
        return

    # Split into chunks that fit on pages
    wrapper = textwrap.TextWrapper(width=95)
    lines = text.splitlines()
    wrapped_lines: List[str] = []
    for line in lines:
        if line.strip() == "":
            wrapped_lines.append("")
        elif line.startswith("#"):
            wrapped_lines.append(line)
        else:
            wrapped_lines.extend(wrapper.wrap(line))

    lines_per_page = 52
    for page_start in range(0, len(wrapped_lines), lines_per_page):
        chunk = wrapped_lines[page_start:page_start + lines_per_page]
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        if page_start == 0:
            ax.set_title("Gap Analysis", fontsize=14, fontweight="bold", pad=15)

        page_text = "\n".join(chunk)
        ax.text(0.03, 0.95, page_text, transform=ax.transAxes,
                fontsize=7.5, fontfamily="monospace",
                verticalalignment="top", horizontalalignment="left")

        pdf.savefig(fig)
        plt.close(fig)


def _page_insights(pdf: PdfPages, runs_dir: Path) -> None:
    """Render missed_games_insights.md if it exists."""
    insights_path = runs_dir / "missed_games_insights.md"
    if not insights_path.exists():
        return

    text = insights_path.read_text(encoding="utf-8")
    if not text.strip():
        return

    wrapper = textwrap.TextWrapper(width=95)
    lines = text.splitlines()
    wrapped_lines: List[str] = []
    for line in lines:
        if line.strip() == "":
            wrapped_lines.append("")
        elif line.startswith("#") or line.startswith("|"):
            wrapped_lines.append(line)
        else:
            wrapped_lines.extend(wrapper.wrap(line))

    lines_per_page = 52
    for page_start in range(0, len(wrapped_lines), lines_per_page):
        chunk = wrapped_lines[page_start:page_start + lines_per_page]
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        if page_start == 0:
            ax.set_title("Missed Games Insights", fontsize=14,
                         fontweight="bold", pad=15)

        page_text = "\n".join(chunk)
        ax.text(0.03, 0.95, page_text, transform=ax.transAxes,
                fontsize=7.5, fontfamily="monospace",
                verticalalignment="top", horizontalalignment="left")

        pdf.savefig(fig)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_report(output_root: Path) -> Path:
    """Generate the full PDF report and return its path."""
    runs_dir = output_root.resolve()
    all_metrics = _load_game_metrics(runs_dir)

    if not all_metrics:
        raise SystemExit(f"No metrics.json files found under {runs_dir}")

    pdf_path = runs_dir / "full_report.pdf"
    print(f"Generating report with {len(all_metrics)} games -> {pdf_path}")

    with PdfPages(str(pdf_path)) as pdf:
        # 1. Title page
        _page_title(pdf, all_metrics)

        # 2. Summary table
        _page_summary_table(pdf, all_metrics)

        # 3. Per-game pages (alphabetical)
        for metrics in sorted(all_metrics, key=lambda m: m.get("game", "")):
            _page_game_header(pdf, metrics)

        # 4. Cross-game comparison
        _page_cross_game(pdf, all_metrics)

        # 5. Gap analysis (if exists)
        _page_gap_analysis(pdf, runs_dir)

        # 6. Insights (if exists)
        _page_insights(pdf, runs_dir)

    print(f"Report written: {pdf_path} ({pdf_path.stat().st_size / 1024:.0f} KB)")
    return pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a comprehensive PDF report from game run results.",
    )
    parser.add_argument(
        "--output-root", type=str, default="runs",
        help="Root directory containing per-game run folders (default: runs)",
    )
    args = parser.parse_args()
    generate_report(Path(args.output_root))


if __name__ == "__main__":
    main()

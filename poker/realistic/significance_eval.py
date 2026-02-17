#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path('/Users/sankalp/Projects/experiment/games/poker/realistic')
PYTHON = Path('/Users/sankalp/Projects/experiment/games/poker/.venv/bin/python')
RUNNER = ROOT / 'run_experiment.py'
OUT_PATH = ROOT / 'results' / 'latest_results.json'
SUMMARY_PATH = ROOT / 'results' / 'significance_summary.json'

SEEDS = list(range(1, 31))
TRAIN_STEPS = 10_000
EVAL_HANDS = 200


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def two_sided_p_from_z(z: float) -> float:
    return 2.0 * (1.0 - normal_cdf(abs(z)))


def binom_two_sided_p(k: int, n: int, p0: float = 0.5) -> float:
    probs = [math.comb(n, i) * (p0 ** i) * ((1 - p0) ** (n - i)) for i in range(n + 1)]
    observed = probs[k]
    p = sum(v for v in probs if v <= observed + 1e-12)
    return min(1.0, p)


def quantile(xs, q):
    ys = sorted(xs)
    if not ys:
        return float('nan')
    pos = (len(ys) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ys[lo]
    w = pos - lo
    return ys[lo] * (1 - w) + ys[hi] * w


def bootstrap_ci(values, n_boot=10000):
    import random

    rng = random.Random(12345)
    n = len(values)
    if n == 0:
        return (float('nan'), float('nan'))
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    return quantile(means, 0.025), quantile(means, 0.975)


all_rows = []
profile_to_vals = defaultdict(list)
profile_to_win = defaultdict(list)
profile_to_show = defaultdict(list)

for seed in SEEDS:
    cmd = [
        str(PYTHON),
        str(RUNNER),
        '--seed',
        str(seed),
        '--train-steps',
        str(TRAIN_STEPS),
        '--eval-hands',
        str(EVAL_HANDS),
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        print(f'Run failed for seed={seed}', file=sys.stderr)
        print(proc.stdout, file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        sys.exit(proc.returncode)

    data = json.loads(OUT_PATH.read_text())
    for row in data['results']:
        profile = row['profile']
        bb100 = float(row['bb_per_100'])
        win_rate = float(row['win_rate'])
        showdown_rate = float(row['showdown_rate'])
        profile_to_vals[profile].append(bb100)
        profile_to_win[profile].append(win_rate)
        profile_to_show[profile].append(showdown_rate)
        all_rows.append({'seed': seed, 'profile': profile, 'bb_per_100': bb100, 'win_rate': win_rate, 'showdown_rate': showdown_rate})


summary = {
    'config': {
        'seeds': SEEDS,
        'n_seeds': len(SEEDS),
        'train_steps': TRAIN_STEPS,
        'eval_hands': EVAL_HANDS,
    },
    'per_profile': {},
    'pairwise_vs_balanced': {},
    'raw': all_rows,
}

for profile, vals in sorted(profile_to_vals.items()):
    n = len(vals)
    mean = statistics.fmean(vals)
    sd = statistics.stdev(vals) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    z = mean / se if se > 0 else float('inf')
    p = two_sided_p_from_z(z) if math.isfinite(z) else 0.0
    ci_low = mean - 1.96 * se
    ci_high = mean + 1.96 * se
    b_low, b_high = bootstrap_ci(vals, n_boot=5000)

    wins = sum(1 for v in vals if v > 0)
    sign_p = binom_two_sided_p(wins, n, p0=0.5)

    summary['per_profile'][profile] = {
        'n': n,
        'mean_bb_per_100': mean,
        'sd_bb_per_100': sd,
        'se_bb_per_100': se,
        'ci95_norm': [ci_low, ci_high],
        'ci95_bootstrap': [b_low, b_high],
        'z_against_0': z,
        'p_two_sided_norm': p,
        'wins_positive_seeds': wins,
        'sign_test_p_two_sided': sign_p,
        'mean_win_rate': statistics.fmean(profile_to_win[profile]),
        'mean_showdown_rate': statistics.fmean(profile_to_show[profile]),
    }

if 'balanced' in profile_to_vals:
    base = profile_to_vals['balanced']
    for profile, vals in sorted(profile_to_vals.items()):
        if profile == 'balanced':
            continue
        diffs = [v - b for v, b in zip(vals, base)]
        n = len(diffs)
        mean = statistics.fmean(diffs)
        sd = statistics.stdev(diffs) if n > 1 else 0.0
        se = sd / math.sqrt(n) if n > 1 else 0.0
        z = mean / se if se > 0 else float('inf')
        p = two_sided_p_from_z(z) if math.isfinite(z) else 0.0
        summary['pairwise_vs_balanced'][profile] = {
            'mean_diff_bb_per_100': mean,
            'ci95_norm': [mean - 1.96 * se, mean + 1.96 * se],
            'p_two_sided_norm': p,
        }

SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
print(f'Saved {SUMMARY_PATH}')
for profile, obj in summary['per_profile'].items():
    lo, hi = obj['ci95_bootstrap']
    print(
        f"{profile:16s} mean={obj['mean_bb_per_100']:8.2f} bb/100 "
        f"boot95=[{lo:8.2f}, {hi:8.2f}] p≈{obj['p_two_sided_norm']:.3g} "
        f"sign_p={obj['sign_test_p_two_sided']:.3g}"
    )

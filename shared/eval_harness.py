from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

from shared.game_specs import GameSpec, get_game_spec
from shared.methods import ALGORITHM_NAMES, ActionDecision, StrategyLearner, build_method
from shared.mock_video_feed import TELL_DIM, MockVideoFeed
from shared.opponent_intel import PROFILE_TRAIT_NAMES, OpponentProfile, choose_profile, profile_to_dict, profile_to_vector


def _stable_seed(*parts: object) -> int:
    hasher = hashlib.blake2s(digest_size=4)
    for part in parts:
        hasher.update(str(part).encode("utf-8"))
        hasher.update(b"|")
    return int.from_bytes(hasher.digest(), byteorder="big", signed=False)


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0)))


def two_sided_p_from_z(z_value: float) -> float:
    if not math.isfinite(z_value):
        return 0.0
    return 2.0 * (1.0 - normal_cdf(abs(z_value)))


def summarize(values: Sequence[float]) -> Dict[str, float | int | List[float]]:
    n = len(values)
    if n == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "sd": float("nan"),
            "se": float("nan"),
            "ci95": [float("nan"), float("nan")],
            "z_vs_zero": float("nan"),
            "p_vs_zero": float("nan"),
        }

    mean = float(statistics.fmean(values))
    sd = float(statistics.stdev(values)) if n > 1 else 0.0
    se = float(sd / math.sqrt(n)) if n > 1 else 0.0
    ci95 = [mean - 1.96 * se, mean + 1.96 * se]

    if se > 0.0:
        z_value = mean / se
        p_value = two_sided_p_from_z(z_value)
    elif mean == 0.0:
        z_value = 0.0
        p_value = 1.0
    else:
        z_value = float("inf") if mean > 0.0 else float("-inf")
        p_value = 0.0

    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "se": se,
        "ci95": [float(ci95[0]), float(ci95[1])],
        "z_vs_zero": float(z_value),
        "p_vs_zero": float(p_value),
    }


def summarize_paired(diffs: Sequence[float]) -> Dict[str, float | int | List[float]]:
    stats = summarize(diffs)
    return {
        "n": int(stats["n"]),
        "mean_diff": float(stats["mean"]),
        "sd_diff": float(stats["sd"]),
        "se_diff": float(stats["se"]),
        "ci95": list(stats["ci95"]),
        "z_vs_zero": float(stats["z_vs_zero"]),
        "p_vs_zero": float(stats["p_vs_zero"]),
    }


class LightweightGameSimulator:
    def __init__(
        self,
        spec: GameSpec,
        profile: OpponentProfile,
        seed: int,
        tell_scale: float = 1.0,
    ) -> None:
        self.spec = spec
        self.profile = profile
        self.seed = int(seed)
        self.tell_scale = float(tell_scale)

        self.profile_vector = profile_to_vector(profile)
        self.state_dim = self.spec.state_dim(profile_dim=len(self.profile_vector))
        self.base_dim = self.spec.base_state_dim
        self.tell_start = self.base_dim
        self.tell_end = self.tell_start + TELL_DIM

        self.feed = MockVideoFeed(profile=profile, seed=self.seed + 17, interval_seconds=10)

        rng = np.random.default_rng(_stable_seed(self.spec.name, self.seed, "action_weights"))
        self.action_weights = rng.normal(0.0, 1.0, size=(self.spec.n_actions, self.state_dim)).astype(np.float64)
        self.action_bias = rng.normal(0.0, 0.06, size=(self.spec.n_actions,)).astype(np.float64)

        self.action_weights[:, : self.base_dim] *= 0.70 + 0.30 * self.spec.difficulty
        self.action_weights[:, self.tell_start : self.tell_end] *= 0.70 + self.spec.tell_weight
        self.action_weights[:, self.tell_end :] *= 0.70 + self.spec.profile_weight
        self.action_weights /= math.sqrt(max(1, self.state_dim))

        self._step = 0
        self._current_state: np.ndarray | None = None

    def _rng_for_step(self, step: int, salt: str) -> np.random.Generator:
        return np.random.default_rng(_stable_seed(self.spec.name, self.seed, step, salt))

    def _build_base_state(self, step: int) -> np.ndarray:
        rng = self._rng_for_step(step, "base")
        base = rng.normal(0.0, 1.0, size=(self.base_dim,)).astype(np.float64)

        idx = np.arange(1, self.base_dim + 1, dtype=np.float64)
        phase = np.sin((step + 1) * idx * (0.03 + 0.005 * self.spec.difficulty))
        pace = np.cos((step + 1) * idx * 0.017)

        base = 0.60 * base + 0.25 * phase + 0.15 * pace
        base[0] += (self.profile.aggression - self.profile.tightness) * 0.85
        base[1] += (self.profile.bluff_rate - 0.50) * 0.70
        base[2] += (self.profile.patience - 0.50) * 0.60
        return base.astype(np.float32)

    def _build_state(self, step: int) -> np.ndarray:
        base_state = self._build_base_state(step)
        tell = self.feed.tell_for_step(step_index=step, step_seconds=self.spec.step_seconds)
        tell = np.clip(tell * self.tell_scale, 0.0, 1.5).astype(np.float32)
        return np.concatenate([base_state, tell, self.profile_vector]).astype(np.float32)

    def reset(self) -> np.ndarray:
        self._step = 0
        self._current_state = self._build_state(self._step)
        return self._current_state.copy()

    def _expected_rewards(self, state: np.ndarray) -> np.ndarray:
        state64 = state.astype(np.float64)
        base_scores = self.action_weights @ state64 + self.action_bias

        tell = state64[self.tell_start : self.tell_end]
        profile = state64[self.tell_end :]

        stress = float(np.mean(tell[:6]))
        confidence = float(np.mean(tell[6:]))
        aggression = float(profile[0])
        tightness = float(profile[1])

        action_axis = np.linspace(-0.35, 0.35, self.spec.n_actions)
        style_term = action_axis * (
            0.60 * confidence
            - 0.45 * stress
            + 0.35 * aggression
            - 0.25 * tightness
        )

        signal_scale = 1.25 - 0.45 * self.spec.difficulty
        logits = signal_scale * (base_scores + style_term)
        return logits - float(np.mean(logits))

    def step(self, action: int) -> tuple[np.ndarray, float, bool, Dict[str, float | int]]:
        if self._current_state is None:
            self.reset()
        assert self._current_state is not None

        action = int(np.clip(action, 0, self.spec.n_actions - 1))
        expected = self._expected_rewards(self._current_state)
        optimal_action = int(np.argmax(expected))

        noise_rng = self._rng_for_step(self._step, "noise")
        noise = noise_rng.normal(0.0, self.spec.reward_noise, size=(self.spec.n_actions,))
        margin = float(expected[optimal_action] - expected[action])

        # Dense supervision keeps training stable in low-sample simulation:
        # choose the oracle action -> positive reward, otherwise penalize by regret margin.
        classification_reward = 1.0 if action == optimal_action else (-0.35 - 0.15 * margin)
        signal_reward = 0.25 * float(expected[action])
        reward = float(classification_reward + signal_reward + noise[action])

        self._step += 1
        done = self._step >= self.spec.episode_steps
        next_state = self._build_state(0 if done else self._step)
        self._current_state = next_state

        info: Dict[str, float | int] = {
            "optimal_action": optimal_action,
            "oracle_reward": float(np.max(expected)),
            "expected_reward": float(expected[action]),
        }
        return next_state.copy(), reward, done, info


def train_method(
    method: StrategyLearner,
    spec: GameSpec,
    profile: OpponentProfile,
    seed: int,
    train_steps: int,
) -> None:
    env = LightweightGameSimulator(
        spec=spec,
        profile=profile,
        seed=_stable_seed(spec.name, seed, "train_env"),
        tell_scale=1.0,
    )
    state = env.reset()
    for _ in range(max(1, int(train_steps))):
        decision: ActionDecision = method.act(state, explore=True)
        next_state, reward, done, _ = env.step(decision.action)
        method.learn(
            state=state,
            action=decision.action,
            reward=reward,
            next_state=next_state,
            done=done,
            decision=decision,
        )
        state = env.reset() if done else next_state


def evaluate_method(
    method: StrategyLearner,
    spec: GameSpec,
    profile: OpponentProfile,
    seed: int,
    eval_steps: int,
    tell_scale: float,
) -> Dict[str, float]:
    env = LightweightGameSimulator(
        spec=spec,
        profile=profile,
        seed=_stable_seed(spec.name, seed, "eval_env", tell_scale),
        tell_scale=tell_scale,
    )
    state = env.reset()

    rewards: List[float] = []
    optimal_hits = 0

    for _ in range(max(1, int(eval_steps))):
        decision = method.act(state, explore=False)
        next_state, reward, done, info = env.step(decision.action)
        rewards.append(float(reward))
        optimal_hits += int(decision.action == int(info["optimal_action"]))
        state = env.reset() if done else next_state

    n = len(rewards)
    total_reward = float(np.sum(rewards))
    return {
        "mean_reward": float(total_reward / max(1, n)),
        "total_reward": total_reward,
        "optimal_action_rate": float(optimal_hits / max(1, n)),
    }


def run_game_protocol(
    game_name: str,
    algorithms: Sequence[str],
    seeds: Sequence[int],
    train_steps: int,
    eval_steps: int,
    tell_scales: Sequence[float],
) -> Dict[str, object]:
    spec = get_game_spec(game_name)

    algorithms_clean = [name.strip().upper() for name in algorithms if name.strip()]
    for name in algorithms_clean:
        if name not in ALGORITHM_NAMES:
            raise ValueError(f"Unsupported algorithm '{name}'. Allowed: {', '.join(ALGORITHM_NAMES)}")

    profile_dim = len(PROFILE_TRAIT_NAMES)
    state_dim = spec.state_dim(profile_dim=profile_dim)

    rewards_by_algorithm: Dict[str, List[float]] = {algo: [] for algo in algorithms_clean}
    seed_matrix: Dict[int, Dict[str, float]] = {}
    seed_rows: List[Dict[str, object]] = []

    sensitivity_eval_steps = max(40, int(eval_steps) // 2)
    tell_raw: List[Dict[str, object]] = []

    for seed in seeds:
        profile_name, profile = choose_profile(spec.name, int(seed))
        seed_matrix[int(seed)] = {}

        for algorithm in algorithms_clean:
            method = build_method(
                name=algorithm,
                state_dim=state_dim,
                n_actions=spec.n_actions,
                seed=_stable_seed(spec.name, seed, algorithm, "method"),
            )
            train_method(
                method=method,
                spec=spec,
                profile=profile,
                seed=int(seed),
                train_steps=int(train_steps),
            )
            eval_metrics = evaluate_method(
                method=method,
                spec=spec,
                profile=profile,
                seed=int(seed),
                eval_steps=int(eval_steps),
                tell_scale=1.0,
            )

            rewards_by_algorithm[algorithm].append(float(eval_metrics["mean_reward"]))
            seed_matrix[int(seed)][algorithm] = float(eval_metrics["mean_reward"])

            seed_rows.append(
                {
                    "seed": int(seed),
                    "algorithm": algorithm,
                    "profile_name": profile_name,
                    "profile_traits": profile_to_dict(profile),
                    "mean_reward": float(eval_metrics["mean_reward"]),
                    "total_reward": float(eval_metrics["total_reward"]),
                    "optimal_action_rate": float(eval_metrics["optimal_action_rate"]),
                }
            )

            for scale in tell_scales:
                sens_metrics = evaluate_method(
                    method=method,
                    spec=spec,
                    profile=profile,
                    seed=int(seed),
                    eval_steps=sensitivity_eval_steps,
                    tell_scale=float(scale),
                )
                tell_raw.append(
                    {
                        "seed": int(seed),
                        "algorithm": algorithm,
                        "tell_scale": float(scale),
                        "mean_reward": float(sens_metrics["mean_reward"]),
                    }
                )

    per_algorithm_stats: Dict[str, Dict[str, float | int | List[float]]] = {}
    for algorithm in algorithms_clean:
        per_algorithm_stats[algorithm] = summarize(rewards_by_algorithm[algorithm])

    pairwise: Dict[str, Dict[str, float | int | List[float]]] = {}
    for left, right in itertools.combinations(algorithms_clean, 2):
        diffs = [seed_matrix[int(seed)][left] - seed_matrix[int(seed)][right] for seed in seeds]
        pairwise[f"{left}_minus_{right}"] = summarize_paired(diffs)

    tell_summary: Dict[str, List[Dict[str, float | int | List[float]]]] = {}
    for algorithm in algorithms_clean:
        by_scale: Dict[float, List[float]] = defaultdict(list)
        for row in tell_raw:
            if row["algorithm"] == algorithm:
                by_scale[float(row["tell_scale"])].append(float(row["mean_reward"]))
        tell_summary[algorithm] = []
        for scale in sorted(by_scale):
            stats = summarize(by_scale[scale])
            tell_summary[algorithm].append(
                {
                    "tell_scale": float(scale),
                    "n": int(stats["n"]),
                    "mean": float(stats["mean"]),
                    "sd": float(stats["sd"]),
                    "se": float(stats["se"]),
                    "ci95": list(stats["ci95"]),
                    "z_vs_zero": float(stats["z_vs_zero"]),
                    "p_vs_zero": float(stats["p_vs_zero"]),
                }
            )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "game": spec.name,
        "config": {
            "algorithms": algorithms_clean,
            "seed_start": int(seeds[0]) if seeds else None,
            "seeds": [int(seed) for seed in seeds],
            "n_seeds": len(seeds),
            "train_steps": int(train_steps),
            "eval_steps": int(eval_steps),
            "sensitivity_eval_steps": int(sensitivity_eval_steps),
            "tell_scales": [float(scale) for scale in tell_scales],
        },
        "state_schema": {
            "base_state_dim": spec.base_state_dim,
            "tell_dim": TELL_DIM,
            "profile_trait_names": list(PROFILE_TRAIT_NAMES),
            "state_dim": state_dim,
            "tell_interval_seconds": 10,
            "game_spec": spec.to_dict(),
        },
        "per_algorithm": per_algorithm_stats,
        "pairwise": pairwise,
        "seed_results": seed_rows,
        "tell_sensitivity": {
            "per_algorithm": tell_summary,
            "raw": tell_raw,
        },
    }


def save_metrics(metrics: Mapping[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")


def topline_from_metrics(metrics: Mapping[str, object]) -> Dict[str, object]:
    per_algorithm = metrics.get("per_algorithm", {})
    best_algorithm = None
    best_mean = float("-inf")

    for algorithm, stats in per_algorithm.items():
        mean = float(stats.get("mean", float("-inf")))
        if mean > best_mean:
            best_mean = mean
            best_algorithm = algorithm

    return {
        "game": metrics.get("game"),
        "best_algorithm": best_algorithm,
        "best_mean_reward": best_mean,
    }

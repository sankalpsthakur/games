#!/usr/bin/env python3
"""Train/evaluate PPO in a realistic-ish heads-up Hold'em environment.

This script wires:
- realistic state/action/outcome spaces from `holdem_env.py`
- mock video-derived opponent features every 10 seconds from `opponent_intel.py`
- cross-profile evaluation with bankroll-style diagnostics
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm

from holdem_env import HoldemEnv
from opponent_intel import DEFAULT_PROFILES, MockVideoFeed, OpponentIntelModel, PlayerProfile

RESULTS_PATH = Path(
    "/Users/sankalp/Projects/experiment/games/poker/realistic/results/latest_results.json"
)

AGGRESSIVE_ACTIONS = {"bet_half_pot", "bet_pot", "raise_2x_pot", "all_in"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realistic Hold'em multi-method experiment")
    parser.add_argument("--algorithm", type=str, default="PPO", choices=["PPO", "A2C", "DQN"])
    parser.add_argument("--train-profile", type=str, default="balanced")
    parser.add_argument("--train-steps", type=int, default=10_000)
    parser.add_argument("--eval-hands", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--results-path", type=Path, default=RESULTS_PATH)
    return parser.parse_args()


class IntelDrivenOpponent:
    """Adapter that maps mock video+profile signals to opponent policy outputs."""

    def __init__(self, profile_name: str, profile: PlayerProfile, seed: int) -> None:
        self.profile_name = profile_name
        self.profile = profile
        self.model = OpponentIntelModel()
        self.feed = MockVideoFeed(profile=profile, seed=seed, interval_s=10)

    def __call__(
        self,
        state: Dict[str, Any],
        legal_mask: np.ndarray,
        rng: np.random.Generator,
    ) -> Dict[str, Any]:
        del legal_mask, rng  # Environment handles legal-action filtering and sampling.

        tell = self.feed.snapshot_at(float(state.get("timestamp_s", 0.0)))
        distribution = self.model.estimate_next_move_distribution(
            state=state,
            profile=self.profile,
            tell=tell,
        )
        # 9 features: 5 tell features + 4 profile traits.
        opponent_intel = np.array(
            [
                tell.gaze_aversion,
                tell.blink_rate,
                tell.jaw_tension,
                tell.voice_stress,
                tell.confidence,
                self.profile.aggression,
                self.profile.tightness,
                self.profile.bluff_freq,
                self.profile.patience,
            ],
            dtype=np.float32,
        )
        return {"distribution": distribution, "opponent_intel": opponent_intel}


def make_env(profile_name: str, seed: int) -> HoldemEnv:
    profile = DEFAULT_PROFILES[profile_name]
    env = HoldemEnv(
        starting_stack=100.0,
        small_blind=0.5,
        big_blind=1.0,
        agent_seat=0,
        opponent_intel_size=9,
    )
    env.configure_opponent(
        profile_name=profile_name,
        profile=profile,
        callback=IntelDrivenOpponent(profile_name, profile, seed=seed),
    )
    env.reset(seed=seed)
    return env


def _as_action_int(action: Any) -> int:
    if isinstance(action, np.ndarray):
        return int(action.squeeze().item())
    return int(action)


def evaluate_profile(
    model: BaseAlgorithm,
    profile_name: str,
    eval_hands: int,
    seed: int,
) -> Dict[str, Any]:
    env = make_env(profile_name, seed)

    rewards: list[float] = []
    bankroll_path: list[float] = [0.0]
    wins = 0
    showdowns = 0
    total_pot_bb = 0.0
    fold_count = 0
    call_count = 0
    aggressive_count = 0
    decision_count = 0

    for hand_idx in range(eval_hands):
        obs, _ = env.reset(seed=seed + hand_idx)
        terminated = False
        truncated = False
        hand_reward = 0.0
        last_info: Mapping[str, Any] = {}

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            action_int = _as_action_int(action)
            obs, reward, terminated, truncated, info = env.step(action_int)
            hand_reward += float(reward)
            last_info = info

            action_name = str(info.get("resolved_action_name", ""))
            if action_name:
                decision_count += 1
                if action_name == "fold":
                    fold_count += 1
                elif action_name == "call":
                    call_count += 1
                elif action_name in AGGRESSIVE_ACTIONS:
                    aggressive_count += 1

        rewards.append(hand_reward)
        bankroll_path.append(bankroll_path[-1] + hand_reward)
        wins += int(bool(last_info.get("won_hand", False)))
        showdowns += int(bool(last_info.get("showdown", False)))
        total_pot_bb += float(last_info.get("pot_bb", 0.0))

    env.close()

    n = max(1, len(rewards))
    mean_reward = float(np.mean(rewards)) if rewards else 0.0
    bb_per_100 = float(np.sum(rewards) / n * 100.0)
    peak = float("-inf")
    max_drawdown = 0.0
    for value in bankroll_path:
        peak = max(peak, value)
        max_drawdown = max(max_drawdown, peak - value)

    return {
        "profile": profile_name,
        "mean_reward_bb": mean_reward,
        "win_rate": float(wins / n),
        "showdown_rate": float(showdowns / n),
        "avg_pot_bb": float(total_pot_bb / n),
        "bb_per_100": bb_per_100,
        "max_drawdown_bb": float(max_drawdown),
        "fold_rate": float(fold_count / decision_count) if decision_count else 0.0,
        "call_rate": float(call_count / decision_count) if decision_count else 0.0,
        "aggression_rate": float(aggressive_count / decision_count) if decision_count else 0.0,
    }


def summarize_table(rows: list[Dict[str, Any]]) -> str:
    headers = [
        "profile",
        "mean_reward_bb",
        "win_rate",
        "showdown_rate",
        "avg_pot_bb",
        "bb_per_100",
        "max_dd_bb",
    ]
    table = [headers]
    for row in rows:
        table.append(
            [
                row["profile"],
                f"{row['mean_reward_bb']:.3f}",
                f"{row['win_rate'] * 100:.1f}%",
                f"{row['showdown_rate'] * 100:.1f}%",
                f"{row['avg_pot_bb']:.2f}",
                f"{row['bb_per_100']:.1f}",
                f"{row['max_drawdown_bb']:.1f}",
            ]
        )

    widths = [max(len(r[col]) for r in table) for col in range(len(headers))]

    def fmt_line(values: list[str]) -> str:
        return " | ".join(v.ljust(widths[idx]) for idx, v in enumerate(values))

    out = [fmt_line(table[0]), "-+-".join("-" * w for w in widths)]
    out.extend(fmt_line(row) for row in table[1:])
    return "\n".join(out)


def infer_insights(rows: list[Dict[str, Any]]) -> list[str]:
    if not rows:
        return ["No rows produced."]

    ordered = sorted(rows, key=lambda row: row["bb_per_100"], reverse=True)
    easiest = ordered[0]
    hardest = ordered[-1]

    insights = [
        (
            f"Easiest profile: {easiest['profile']} "
            f"({easiest['bb_per_100']:.1f} bb/100, win rate {easiest['win_rate'] * 100:.1f}%)."
        ),
        (
            f"Hardest profile: {hardest['profile']} "
            f"({hardest['bb_per_100']:.1f} bb/100, win rate {hardest['win_rate'] * 100:.1f}%)."
        ),
    ]

    overfold = [r["profile"] for r in rows if r["bb_per_100"] < 0 and r["fold_rate"] >= 0.40]
    overcall = [r["profile"] for r in rows if r["bb_per_100"] < 0 and r["call_rate"] >= 0.52]
    passive = [
        r["profile"] for r in rows if r["bb_per_100"] < 0 and r["aggression_rate"] <= 0.18
    ]

    if overfold:
        insights.append(
            f"Likely overfolding against: {', '.join(overfold)}. Consider defending wider in medium-pot spots."
        )
    if overcall:
        insights.append(
            f"Likely overcalling against: {', '.join(overcall)}. Tighten bluff-catch thresholds."
        )
    if passive:
        insights.append(
            f"Likely under-bluffing / too passive versus: {', '.join(passive)}."
        )
    if not overfold and not overcall and not passive:
        insights.append("No strong leak signature found; increase eval hands for sharper inference.")

    return insights


def _profile_to_jsonable(profile: Any) -> Any:
    if is_dataclass(profile):
        return asdict(profile)
    return profile


def build_model(
    algorithm: str,
    env: HoldemEnv,
    seed: int,
    device: str,
) -> BaseAlgorithm:
    if algorithm == "PPO":
        return PPO(
            "MlpPolicy",
            env,
            verbose=0,
            seed=seed,
            device=device,
            n_steps=512,
            batch_size=128,
            gamma=0.995,
        )
    if algorithm == "A2C":
        return A2C(
            "MlpPolicy",
            env,
            verbose=0,
            seed=seed,
            device=device,
            n_steps=10,
            gamma=0.995,
            learning_rate=7e-4,
        )
    if algorithm == "DQN":
        return DQN(
            "MlpPolicy",
            env,
            verbose=0,
            seed=seed,
            device=device,
            gamma=0.995,
            learning_rate=1e-4,
            learning_starts=500,
            buffer_size=50_000,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=500,
        )
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def run_experiment(
    algorithm: str,
    train_profile: str,
    train_steps: int,
    eval_hands: int,
    seed: int,
    device: str,
    print_output: bool = True,
) -> Dict[str, Any]:
    if train_profile not in DEFAULT_PROFILES:
        available = ", ".join(DEFAULT_PROFILES.keys())
        raise ValueError(f"Unknown profile '{train_profile}'. Available: {available}")

    random.seed(seed)
    np.random.seed(seed)

    if print_output:
        print(
            f"Training {algorithm} on '{train_profile}' for {train_steps} timesteps "
            f"(eval {eval_hands} hands/profile)..."
        )

    train_env = make_env(profile_name=train_profile, seed=seed)
    model = build_model(algorithm=algorithm, env=train_env, seed=seed, device=device)
    model.learn(total_timesteps=max(1, int(train_steps)), progress_bar=False)
    train_env.close()

    rows: list[Dict[str, Any]] = []
    for idx, profile_name in enumerate(DEFAULT_PROFILES.keys()):
        row = evaluate_profile(
            model=model,
            profile_name=profile_name,
            eval_hands=max(1, int(eval_hands)),
            seed=seed + (idx + 1) * 10_000,
        )
        rows.append(row)

    insights = infer_insights(rows)
    if print_output:
        print("\nCross-profile summary")
        print(summarize_table(rows))
        print("\nInsights")
        for line in insights:
            print(f"- {line}")

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": algorithm,
        "assumptions": {
            "video_feed": "mocked",
            "tell_update_interval_seconds": 10,
            "profiles": {
                k: _profile_to_jsonable(v)
                for k, v in DEFAULT_PROFILES.items()
            },
            "environment": "heads_up_no_limit_holdem_discrete_abstraction",
        },
        "train_profile": train_profile,
        "train_steps": int(train_steps),
        "eval_hands_per_profile": int(eval_hands),
        "seed": int(seed),
        "results": rows,
        "insights": insights,
    }


def main() -> None:
    args = parse_args()
    payload = run_experiment(
        algorithm=args.algorithm,
        train_profile=args.train_profile,
        train_steps=args.train_steps,
        eval_hands=args.eval_hands,
        seed=args.seed,
        device=args.device,
        print_output=True,
    )

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    with args.results_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved results: {args.results_path}")


if __name__ == "__main__":
    main()

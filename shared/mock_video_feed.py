from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from shared.opponent_intel import OpponentProfile, profile_to_vector

TELL_FEATURE_NAMES: Tuple[str, ...] = (
    "gaze_aversion",
    "blink_rate",
    "jaw_tension",
    "voice_stress",
    "breathing_irregularity",
    "micro_tremor",
    "confidence",
    "composure_shift",
)

TELL_DIM = len(TELL_FEATURE_NAMES)


def _clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


@dataclass(frozen=True)
class VideoTellSnapshot:
    timestamp_seconds: float
    values: np.ndarray

    def as_dict(self) -> Dict[str, float]:
        return {
            "timestamp_seconds": float(self.timestamp_seconds),
            **{name: float(self.values[i]) for i, name in enumerate(TELL_FEATURE_NAMES)},
        }


class MockVideoFeed:
    """Deterministic 8-dim tells sampled on a fixed cadence (default every 10s)."""

    def __init__(
        self,
        profile: OpponentProfile,
        seed: int,
        interval_seconds: int = 10,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError(f"interval_seconds must be > 0, got {interval_seconds}")
        self.profile = profile
        self.seed = int(seed)
        self.interval_seconds = int(interval_seconds)
        profile_vec = profile_to_vector(profile)
        multipliers = np.array([113.0, 211.0, 307.0, 401.0, 503.0], dtype=np.float64)
        self._profile_code = int(np.dot(profile_vec.astype(np.float64), multipliers) * 10_000) % (2**32)

    def _rng_for_index(self, index: int) -> np.random.Generator:
        mixed = (
            self.seed * 2_654_435_761
            + self._profile_code * 97_531
            + int(index) * 67_867
        ) % (2**32)
        return np.random.default_rng(mixed)

    def tell_at(self, timestamp_seconds: float) -> np.ndarray:
        if timestamp_seconds < 0:
            raise ValueError(f"timestamp_seconds must be >= 0, got {timestamp_seconds}")

        index = int(timestamp_seconds // self.interval_seconds)
        rng = self._rng_for_index(index)

        fatigue = _clamp01(index / 70.0)
        volatility = _clamp01(1.0 - self.profile.tilt_resistance)
        pressure = _clamp01(0.50 * volatility + 0.30 * (1.0 - self.profile.patience) + 0.20 * fatigue)

        gaze_aversion = _clamp01(0.20 + 0.25 * pressure + 0.12 * self.profile.bluff_rate + rng.normal(0.0, 0.05))
        blink_rate = _clamp01(0.24 + 0.30 * pressure + 0.10 * fatigue + rng.normal(0.0, 0.05))
        jaw_tension = _clamp01(0.22 + 0.28 * pressure + 0.10 * self.profile.aggression + rng.normal(0.0, 0.05))
        voice_stress = _clamp01(0.18 + 0.35 * pressure + 0.10 * (1.0 - self.profile.tightness) + rng.normal(0.0, 0.05))
        breathing_irregularity = _clamp01(0.16 + 0.30 * pressure + 0.15 * fatigue + rng.normal(0.0, 0.05))
        micro_tremor = _clamp01(0.12 + 0.30 * pressure + 0.18 * (1.0 - self.profile.tilt_resistance) + rng.normal(0.0, 0.05))
        confidence = _clamp01(
            0.35
            + 0.25 * self.profile.aggression
            + 0.20 * self.profile.patience
            - 0.20 * voice_stress
            - 0.08 * gaze_aversion
            + rng.normal(0.0, 0.04)
        )
        composure_shift = _clamp01(0.55 + 0.30 * self.profile.tilt_resistance - 0.28 * pressure + rng.normal(0.0, 0.04))

        return np.array(
            [
                gaze_aversion,
                blink_rate,
                jaw_tension,
                voice_stress,
                breathing_irregularity,
                micro_tremor,
                confidence,
                composure_shift,
            ],
            dtype=np.float32,
        )

    def snapshot_at(self, timestamp_seconds: float) -> VideoTellSnapshot:
        return VideoTellSnapshot(
            timestamp_seconds=float(timestamp_seconds),
            values=self.tell_at(timestamp_seconds),
        )

    def tell_for_step(self, step_index: int, step_seconds: int) -> np.ndarray:
        ts = max(0, int(step_index)) * max(1, int(step_seconds))
        return self.tell_at(float(ts))

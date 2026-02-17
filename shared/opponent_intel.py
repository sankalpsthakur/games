from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple
import zlib

import numpy as np

PROFILE_TRAIT_NAMES: Tuple[str, ...] = (
    "aggression",
    "tightness",
    "bluff_rate",
    "patience",
    "tilt_resistance",
)


def _clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


@dataclass(frozen=True)
class OpponentProfile:
    aggression: float
    tightness: float
    bluff_rate: float
    patience: float
    tilt_resistance: float

    def __post_init__(self) -> None:
        for name in PROFILE_TRAIT_NAMES:
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")


DEFAULT_PROFILES: Dict[str, OpponentProfile] = {
    "balanced": OpponentProfile(0.50, 0.50, 0.50, 0.55, 0.60),
    "aggressive": OpponentProfile(0.82, 0.28, 0.70, 0.32, 0.42),
    "tight": OpponentProfile(0.28, 0.84, 0.22, 0.74, 0.76),
    "tricky": OpponentProfile(0.63, 0.40, 0.86, 0.45, 0.55),
    "composed": OpponentProfile(0.44, 0.62, 0.35, 0.80, 0.88),
}


def profile_to_vector(profile: OpponentProfile) -> np.ndarray:
    return np.array([getattr(profile, name) for name in PROFILE_TRAIT_NAMES], dtype=np.float32)


def profile_to_dict(profile: OpponentProfile) -> Dict[str, float]:
    data = asdict(profile)
    return {key: float(value) for key, value in data.items()}


def _stable_u32(text: str) -> int:
    return int(zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF)


def choose_profile(game_name: str, seed: int) -> tuple[str, OpponentProfile]:
    profile_names = sorted(DEFAULT_PROFILES)
    index = (_stable_u32(game_name.lower()) + int(seed) * 1_315_423_911) % len(profile_names)
    name = profile_names[int(index)]
    return name, DEFAULT_PROFILES[name]


def blend_profiles(
    left: OpponentProfile,
    right: OpponentProfile,
    weight_right: float,
) -> OpponentProfile:
    wr = _clamp01(weight_right)
    wl = 1.0 - wr
    return OpponentProfile(
        aggression=_clamp01(wl * left.aggression + wr * right.aggression),
        tightness=_clamp01(wl * left.tightness + wr * right.tightness),
        bluff_rate=_clamp01(wl * left.bluff_rate + wr * right.bluff_rate),
        patience=_clamp01(wl * left.patience + wr * right.patience),
        tilt_resistance=_clamp01(wl * left.tilt_resistance + wr * right.tilt_resistance),
    )

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Mapping

import numpy as np

ACTIONS: tuple[str, ...] = (
    "fold",
    "check",
    "call",
    "bet_small",
    "bet_pot",
    "raise",
    "all_in",
)


def _clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _validate_unit_interval(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


@dataclass(frozen=True)
class PlayerProfile:
    aggression: float
    tightness: float
    bluff_freq: float
    patience: float
    tilt_resistance: float

    def __post_init__(self) -> None:
        _validate_unit_interval("aggression", self.aggression)
        _validate_unit_interval("tightness", self.tightness)
        _validate_unit_interval("bluff_freq", self.bluff_freq)
        _validate_unit_interval("patience", self.patience)
        _validate_unit_interval("tilt_resistance", self.tilt_resistance)


@dataclass(frozen=True)
class VideoTellSnapshot:
    timestamp_s: float
    gaze_aversion: float
    blink_rate: float
    jaw_tension: float
    voice_stress: float
    confidence: float

    def __post_init__(self) -> None:
        if self.timestamp_s < 0:
            raise ValueError(f"timestamp_s must be >= 0, got {self.timestamp_s}")

        _validate_unit_interval("gaze_aversion", self.gaze_aversion)
        _validate_unit_interval("blink_rate", self.blink_rate)
        _validate_unit_interval("jaw_tension", self.jaw_tension)
        _validate_unit_interval("voice_stress", self.voice_stress)
        _validate_unit_interval("confidence", self.confidence)


class MockVideoFeed:
    """Generate deterministic, profile-shaped tells on a fixed interval."""

    def __init__(
        self,
        profile: PlayerProfile,
        seed: int = 0,
        interval_s: int = 10,
    ) -> None:
        if interval_s <= 0:
            raise ValueError(f"interval_s must be > 0, got {interval_s}")

        self.profile = profile
        self.seed = int(seed)
        self.interval_s = int(interval_s)
        self._next_index = 0
        self._profile_code = self._encode_profile(profile)

    @staticmethod
    def _encode_profile(profile: PlayerProfile) -> int:
        vector = np.array(
            [
                profile.aggression,
                profile.tightness,
                profile.bluff_freq,
                profile.patience,
                profile.tilt_resistance,
            ],
            dtype=np.float64,
        )
        multipliers = np.array([97.0, 193.0, 389.0, 769.0, 1543.0], dtype=np.float64)
        return int(np.dot(vector, multipliers) * 10_000) % (2**32)

    def _rng_for_index(self, index: int) -> np.random.Generator:
        # Mix seed, profile, and time index to keep output deterministic.
        mixed_seed = (
            self.seed * 1_315_423_911
            + self._profile_code * 2_654_435_761
            + index * 97_531
        ) % (2**32)
        return np.random.default_rng(mixed_seed)

    def _snapshot_from_index(self, index: int) -> VideoTellSnapshot:
        rng = self._rng_for_index(index)
        timestamp_s = float(index * self.interval_s)

        fatigue = _clamp01(index / 60.0)
        volatility = 1.0 - self.profile.tilt_resistance

        gaze_aversion = _clamp01(
            0.20
            + 0.35 * volatility
            + 0.15 * self.profile.bluff_freq
            + 0.10 * fatigue
            + rng.normal(0.0, 0.06)
        )
        blink_rate = _clamp01(
            0.25
            + 0.25 * volatility
            + 0.20 * (1.0 - self.profile.patience)
            + 0.20 * fatigue
            + rng.normal(0.0, 0.07)
        )
        jaw_tension = _clamp01(
            0.22
            + 0.30 * volatility
            + 0.20 * self.profile.aggression
            + rng.normal(0.0, 0.08)
        )
        voice_stress = _clamp01(
            0.18
            + 0.35 * volatility
            + 0.25 * fatigue
            + 0.15 * (1.0 - self.profile.tightness)
            + rng.normal(0.0, 0.07)
        )
        confidence = _clamp01(
            0.30
            + 0.35 * self.profile.aggression
            + 0.20 * self.profile.patience
            - 0.25 * voice_stress
            + rng.normal(0.0, 0.05)
        )

        return VideoTellSnapshot(
            timestamp_s=timestamp_s,
            gaze_aversion=gaze_aversion,
            blink_rate=blink_rate,
            jaw_tension=jaw_tension,
            voice_stress=voice_stress,
            confidence=confidence,
        )

    def emit(self) -> VideoTellSnapshot:
        snapshot = self._snapshot_from_index(self._next_index)
        self._next_index += 1
        return snapshot

    def snapshot_at(self, timestamp_s: float) -> VideoTellSnapshot:
        if timestamp_s < 0:
            raise ValueError(f"timestamp_s must be >= 0, got {timestamp_s}")

        index = int(timestamp_s // self.interval_s)
        return self._snapshot_from_index(index)

    def iter_snapshots(
        self,
        duration_s: float,
        start_s: float = 0.0,
    ) -> Iterator[VideoTellSnapshot]:
        if duration_s < 0:
            raise ValueError(f"duration_s must be >= 0, got {duration_s}")
        if start_s < 0:
            raise ValueError(f"start_s must be >= 0, got {start_s}")

        first_index = int(start_s // self.interval_s)
        last_index = int((start_s + duration_s) // self.interval_s)
        for index in range(first_index, last_index + 1):
            yield self._snapshot_from_index(index)


class OpponentIntelModel:
    """Estimate opponent action probabilities from game state + behavioral signals."""

    @staticmethod
    def _state_float(state: Mapping[str, object], key: str, default: float) -> float:
        value = state.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def estimate_next_move_distribution(
        self,
        state: dict,
        profile: PlayerProfile,
        tell: VideoTellSnapshot,
    ) -> Dict[str, float]:
        pot_size = max(1e-6, self._state_float(state, "pot_size", 1.0))
        to_call = max(0.0, self._state_float(state, "to_call", 0.0))
        stack = max(0.0, self._state_float(state, "stack", pot_size * 5.0))

        hand_strength = _clamp01(self._state_float(state, "hand_strength", 0.5))
        position_advantage = _clamp01(
            self._state_float(state, "position_advantage", 0.5)
        )
        board_danger = _clamp01(self._state_float(state, "board_danger", 0.5))

        street = str(state.get("street", "flop")).lower()
        street_factor = {
            "preflop": 0.20,
            "flop": 0.45,
            "turn": 0.70,
            "river": 0.90,
        }.get(street, 0.50)

        facing_bet_raw = state.get("facing_bet", None)
        facing_bet = bool(facing_bet_raw) if facing_bet_raw is not None else (to_call > 0.0)

        pot_pressure = _clamp01(to_call / (pot_size + to_call + 1e-9))
        spr = stack / pot_size
        short_stack_pressure = _clamp01(1.0 / (1.0 + spr))

        tell_stress = _clamp01(
            0.45 * tell.voice_stress + 0.30 * tell.jaw_tension + 0.25 * tell.blink_rate
        )
        tell_confidence = _clamp01(
            0.60 * tell.confidence
            + 0.25 * (1.0 - tell.gaze_aversion)
            + 0.15 * profile.patience
        )

        aggression_drive = _clamp01(profile.aggression * (0.65 + 0.35 * tell_confidence))
        bluff_drive = _clamp01(
            profile.bluff_freq
            * (0.70 + 0.30 * tell_confidence)
            * (1.0 - 0.45 * tell_stress)
        )
        commitment = _clamp01(
            0.55 * hand_strength + 0.25 * tell_confidence + 0.20 * aggression_drive
        )

        fold_score = (
            0.9 * profile.tightness
            + 0.8 * tell_stress
            + 1.2 * pot_pressure
            + 0.4 * board_danger
            - 1.6 * hand_strength
            - 0.7 * tell_confidence
        )
        check_score = (
            0.9 * profile.patience
            + 0.5 * profile.tightness
            + 0.5 * (1.0 - pot_pressure)
            + 0.2 * (1.0 - hand_strength)
            - 0.8 * aggression_drive
        )
        call_score = (
            0.7 * (1.0 - profile.aggression)
            + 0.8 * (1.0 - abs(hand_strength - (1.0 - pot_pressure)))
            + 0.5 * profile.patience
            + 0.2 * tell_confidence
            - 0.3 * tell_stress
        )
        bet_small_score = (
            0.8 * aggression_drive
            + 0.7 * bluff_drive
            + 0.3 * position_advantage
            + 0.3 * (1.0 - board_danger)
            + 0.1 * street_factor
            - 0.4 * pot_pressure
        )
        bet_pot_score = (
            1.0 * aggression_drive
            + 1.0 * hand_strength
            + 0.6 * tell_confidence
            + 0.2 * street_factor
            - 0.6 * tell_stress
            - 0.3 * profile.tightness
        )
        raise_score = (
            0.9 * aggression_drive
            + 0.8 * commitment
            + 0.6 * bluff_drive
            + 0.3 * position_advantage
            - 0.4 * profile.tightness
            - 0.3 * board_danger
        )
        all_in_score = (
            1.4 * commitment
            + 0.9 * short_stack_pressure
            + 0.4 * aggression_drive
            + 0.3 * (1.0 - profile.tilt_resistance) * tell_stress
            + 0.3 * street_factor
            - 0.6 * profile.tightness
        )

        if facing_bet:
            fold_score += 0.4
            call_score += 0.5
            raise_score += 0.6
            check_score -= 2.0
            bet_small_score -= 0.8
            bet_pot_score -= 0.8
        else:
            check_score += 0.8
            bet_small_score += 0.4
            bet_pot_score += 0.3
            fold_score -= 2.0
            call_score -= 1.2
            raise_score -= 0.2

        logits = np.array(
            [
                fold_score,
                check_score,
                call_score,
                bet_small_score,
                bet_pot_score,
                raise_score,
                all_in_score,
            ],
            dtype=np.float64,
        )

        temperature = 0.85 + 0.30 * (1.0 - profile.patience)
        logits = logits / max(temperature, 1e-6)
        logits = logits - float(np.max(logits))

        probs = np.exp(logits)
        probs /= float(np.sum(probs))

        return {action: float(prob) for action, prob in zip(ACTIONS, probs)}


def choose_weighted_action(
    distribution: Mapping[str, float],
    rng: np.random.Generator,
) -> str:
    if not distribution:
        raise ValueError("distribution must be non-empty")

    actions = list(distribution.keys())
    weights = np.array([float(distribution[action]) for action in actions], dtype=np.float64)

    if np.any(weights < 0.0):
        raise ValueError("distribution contains negative probabilities")

    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("distribution must have a positive finite weight sum")

    probs = weights / total
    chosen_index = int(rng.choice(len(actions), p=probs))
    return actions[chosen_index]


DEFAULT_PROFILES: Dict[str, PlayerProfile] = {
    "nit": PlayerProfile(
        aggression=0.20,
        tightness=0.90,
        bluff_freq=0.08,
        patience=0.80,
        tilt_resistance=0.85,
    ),
    "tag": PlayerProfile(
        aggression=0.62,
        tightness=0.72,
        bluff_freq=0.22,
        patience=0.74,
        tilt_resistance=0.78,
    ),
    "lag": PlayerProfile(
        aggression=0.86,
        tightness=0.28,
        bluff_freq=0.48,
        patience=0.42,
        tilt_resistance=0.58,
    ),
    "calling_station": PlayerProfile(
        aggression=0.25,
        tightness=0.18,
        bluff_freq=0.05,
        patience=0.92,
        tilt_resistance=0.45,
    ),
    "balanced": PlayerProfile(
        aggression=0.55,
        tightness=0.50,
        bluff_freq=0.20,
        patience=0.66,
        tilt_resistance=0.72,
    ),
}


__all__ = [
    "ACTIONS",
    "PlayerProfile",
    "VideoTellSnapshot",
    "MockVideoFeed",
    "OpponentIntelModel",
    "choose_weighted_action",
    "DEFAULT_PROFILES",
]

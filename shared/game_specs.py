from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

from shared.mock_video_feed import TELL_DIM


@dataclass(frozen=True)
class GameSpec:
    name: str
    base_state_dim: int
    n_actions: int
    episode_steps: int
    step_seconds: int
    reward_noise: float
    tell_weight: float
    profile_weight: float
    difficulty: float
    missed: bool = False

    def state_dim(self, profile_dim: int) -> int:
        return int(self.base_state_dim + TELL_DIM + profile_dim)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


DEFAULT_GAME_SPECS: Dict[str, GameSpec] = {
    "poker": GameSpec(
        name="poker",
        base_state_dim=12,
        n_actions=7,
        episode_steps=90,
        step_seconds=2,
        reward_noise=0.16,
        tell_weight=0.45,
        profile_weight=0.32,
        difficulty=0.75,
        missed=False,
    ),
    "chess": GameSpec(
        name="chess",
        base_state_dim=14,
        n_actions=10,
        episode_steps=90,
        step_seconds=2,
        reward_noise=0.15,
        tell_weight=0.38,
        profile_weight=0.26,
        difficulty=0.72,
        missed=False,
    ),
    "ultimatum": GameSpec(
        name="ultimatum",
        base_state_dim=9,
        n_actions=11,
        episode_steps=70,
        step_seconds=2,
        reward_noise=0.13,
        tell_weight=0.42,
        profile_weight=0.30,
        difficulty=0.52,
        missed=True,
    ),
    "hawk_dove": GameSpec(
        name="hawk_dove",
        base_state_dim=8,
        n_actions=2,
        episode_steps=70,
        step_seconds=2,
        reward_noise=0.15,
        tell_weight=0.35,
        profile_weight=0.32,
        difficulty=0.57,
        missed=True,
    ),
    "stag_hunt": GameSpec(
        name="stag_hunt",
        base_state_dim=8,
        n_actions=2,
        episode_steps=70,
        step_seconds=2,
        reward_noise=0.14,
        tell_weight=0.36,
        profile_weight=0.31,
        difficulty=0.58,
        missed=True,
    ),
    "signaling": GameSpec(
        name="signaling",
        base_state_dim=10,
        n_actions=6,
        episode_steps=80,
        step_seconds=2,
        reward_noise=0.15,
        tell_weight=0.39,
        profile_weight=0.31,
        difficulty=0.61,
        missed=True,
    ),
    "principal_agent": GameSpec(
        name="principal_agent",
        base_state_dim=11,
        n_actions=8,
        episode_steps=80,
        step_seconds=2,
        reward_noise=0.16,
        tell_weight=0.33,
        profile_weight=0.34,
        difficulty=0.62,
        missed=True,
    ),
    "tenders": GameSpec(
        name="tenders",
        base_state_dim=13,
        n_actions=12,
        episode_steps=90,
        step_seconds=2,
        reward_noise=0.16,
        tell_weight=0.44,
        profile_weight=0.35,
        difficulty=0.70,
        missed=True,
    ),
    "interrogation": GameSpec(
        name="interrogation",
        base_state_dim=12,
        n_actions=12,
        episode_steps=88,
        step_seconds=2,
        reward_noise=0.18,
        tell_weight=0.48,
        profile_weight=0.31,
        difficulty=0.71,
        missed=True,
    ),
    "crisis_negotiations": GameSpec(
        name="crisis_negotiations",
        base_state_dim=12,
        n_actions=9,
        episode_steps=88,
        step_seconds=2,
        reward_noise=0.17,
        tell_weight=0.46,
        profile_weight=0.34,
        difficulty=0.73,
        missed=True,
    ),
    "iterated_prisoners_dilemma": GameSpec(
        name="iterated_prisoners_dilemma",
        base_state_dim=9,
        n_actions=2,
        episode_steps=75,
        step_seconds=2,
        reward_noise=0.14,
        tell_weight=0.34,
        profile_weight=0.31,
        difficulty=0.56,
        missed=True,
    ),
    "public_goods": GameSpec(
        name="public_goods",
        base_state_dim=10,
        n_actions=11,
        episode_steps=78,
        step_seconds=2,
        reward_noise=0.15,
        tell_weight=0.36,
        profile_weight=0.33,
        difficulty=0.60,
        missed=True,
    ),
    "trust_game": GameSpec(
        name="trust_game",
        base_state_dim=9,
        n_actions=11,
        episode_steps=74,
        step_seconds=2,
        reward_noise=0.14,
        tell_weight=0.41,
        profile_weight=0.32,
        difficulty=0.57,
        missed=True,
    ),
    "cournot": GameSpec(
        name="cournot",
        base_state_dim=10,
        n_actions=9,
        episode_steps=76,
        step_seconds=2,
        reward_noise=0.15,
        tell_weight=0.30,
        profile_weight=0.33,
        difficulty=0.63,
        missed=True,
    ),
    "bertrand": GameSpec(
        name="bertrand",
        base_state_dim=10,
        n_actions=9,
        episode_steps=76,
        step_seconds=2,
        reward_noise=0.15,
        tell_weight=0.31,
        profile_weight=0.33,
        difficulty=0.64,
        missed=True,
    ),
    "beer_distribution": GameSpec(
        name="beer_distribution",
        base_state_dim=11,
        n_actions=7,
        episode_steps=84,
        step_seconds=2,
        reward_noise=0.16,
        tell_weight=0.29,
        profile_weight=0.30,
        difficulty=0.66,
        missed=True,
    ),
    "auctions": GameSpec(
        name="auctions",
        base_state_dim=11,
        n_actions=10,
        episode_steps=84,
        step_seconds=2,
        reward_noise=0.16,
        tell_weight=0.40,
        profile_weight=0.35,
        difficulty=0.67,
        missed=True,
    ),
    "market_for_lemons": GameSpec(
        name="market_for_lemons",
        base_state_dim=10,
        n_actions=8,
        episode_steps=80,
        step_seconds=2,
        reward_noise=0.17,
        tell_weight=0.37,
        profile_weight=0.34,
        difficulty=0.68,
        missed=True,
    ),
}


def list_game_names() -> List[str]:
    return sorted(DEFAULT_GAME_SPECS)


def get_game_spec(name: str) -> GameSpec:
    key = name.strip().lower()
    if key not in DEFAULT_GAME_SPECS:
        raise KeyError(f"Unknown game '{name}'. Available: {', '.join(list_game_names())}")
    return DEFAULT_GAME_SPECS[key]


def get_missed_game_names() -> List[str]:
    return [name for name, spec in sorted(DEFAULT_GAME_SPECS.items()) if spec.missed]

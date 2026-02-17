"""Heads-up no-limit Hold'em Gymnasium environment.

This module implements a realistic-ish single-agent environment:
- full 52-card deck
- two private cards each
- flop/turn/river community cards
- betting rounds across preflop/flop/turn/river
- discrete action abstraction with graceful illegal-action remapping

The agent controls a single seat and a built-in heuristic policy controls the
opponent. Episodes are one hand long.
"""

from __future__ import annotations

import itertools
from collections import Counter
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class HoldemEnv(gym.Env):
    """Heads-up no-limit Hold'em with a discrete action abstraction."""

    metadata = {"render_modes": ["human"], "render_fps": 8}

    # Action ids (Discrete(7))
    ACTION_FOLD = 0
    ACTION_CHECK = 1
    ACTION_CALL = 2
    ACTION_BET_HALF_POT = 3
    ACTION_BET_POT = 4
    ACTION_RAISE_2X_POT = 5
    ACTION_ALL_IN = 6
    NUM_ACTIONS = 7

    ACTION_NAMES = {
        ACTION_FOLD: "fold",
        ACTION_CHECK: "check",
        ACTION_CALL: "call",
        ACTION_BET_HALF_POT: "bet_half_pot",
        ACTION_BET_POT: "bet_pot",
        ACTION_RAISE_2X_POT: "raise_2x_pot",
        ACTION_ALL_IN: "all_in",
    }
    ACTION_NAME_TO_ID = {
        "fold": ACTION_FOLD,
        "check": ACTION_CHECK,
        "call": ACTION_CALL,
        "bet_small": ACTION_BET_HALF_POT,
        "bet_half_pot": ACTION_BET_HALF_POT,
        "bet_pot": ACTION_BET_POT,
        "raise": ACTION_RAISE_2X_POT,
        "raise_2x_pot": ACTION_RAISE_2X_POT,
        "all_in": ACTION_ALL_IN,
    }
    ACTION_LABELS = (
        "fold",
        "check",
        "call",
        "bet_half_pot",
        "bet_pot",
        "raise_2x_pot",
        "all_in",
    )

    STREET_PREFLOP = 0
    STREET_FLOP = 1
    STREET_TURN = 2
    STREET_RIVER = 3

    STREET_NAMES = {
        STREET_PREFLOP: "preflop",
        STREET_FLOP: "flop",
        STREET_TURN: "turn",
        STREET_RIVER: "river",
    }

    EPS = 1e-9

    def __init__(
        self,
        *,
        starting_stack: float = 100.0,
        small_blind: float = 0.5,
        big_blind: float = 1.0,
        agent_seat: int = 0,
        opponent_intel_size: int = 4,
        illegal_action_penalty: float = 0.01,
    ) -> None:
        super().__init__()
        if starting_stack <= 0:
            raise ValueError("starting_stack must be > 0")
        if big_blind <= 0 or small_blind <= 0:
            raise ValueError("blinds must be > 0")
        if small_blind >= big_blind:
            raise ValueError("small_blind must be < big_blind")
        if agent_seat not in (0, 1):
            raise ValueError("agent_seat must be 0 or 1")
        if opponent_intel_size < 0:
            raise ValueError("opponent_intel_size must be >= 0")

        self.starting_stack = float(starting_stack)
        self.small_blind = float(small_blind)
        self.big_blind = float(big_blind)
        self.agent_seat = int(agent_seat)
        self.opponent_seat = 1 - self.agent_seat
        self.opponent_intel_size = int(opponent_intel_size)
        self.illegal_action_penalty = float(illegal_action_penalty)

        # Observation layout:
        #  street_one_hot(4)
        #  self_stack_norm, opp_stack_norm, pot_norm, to_call_norm, min_raise_norm (5)
        #  button_indicator(1)
        #  legal_action_mask(7)
        #  hole_strength_proxy(1)
        #  board_texture_features(5)
        #  opponent_intel_placeholder(opponent_intel_size)
        #  expected_next_move_distribution(7)
        self._obs_size = (
            4 + 5 + 1 + self.NUM_ACTIONS + 1 + 5 + self.opponent_intel_size + self.NUM_ACTIONS
        )

        self.action_space = spaces.Discrete(self.NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._obs_size,),
            dtype=np.float32,
        )

        # External hooks (do not import opponent-intel modules directly here).
        self._opponent_intel = np.zeros(self.opponent_intel_size, dtype=np.float32)
        self._expected_next_move = np.zeros(self.NUM_ACTIONS, dtype=np.float32)
        self.opponent_policy_callback: Optional[
            Callable[[Dict[str, Any], np.ndarray, np.random.Generator], Any]
        ] = None
        self.opponent_profile: Any = None
        self.opponent_profile_name: str = "default"
        self.decision_interval_s: float = 10.0
        self.decision_index: int = 0
        self.decision_timestamp_s: float = 0.0

        self.hand_id = 0
        self._init_hand_state()

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def set_external_features(
        self,
        *,
        opponent_intel: Optional[Sequence[float]] = None,
        expected_next_move: Optional[Sequence[float]] = None,
    ) -> None:
        """Set external feature hooks for observation placeholders.

        This keeps the environment extensible without importing any opponent
        intel implementation directly.
        """
        if opponent_intel is not None:
            vec = np.asarray(opponent_intel, dtype=np.float32).flatten()
            out = np.zeros(self.opponent_intel_size, dtype=np.float32)
            n = min(out.size, vec.size)
            if n > 0:
                out[:n] = vec[:n]
            self._opponent_intel = np.clip(out, -1.0, 1.0)

        if expected_next_move is not None:
            vec = np.asarray(expected_next_move, dtype=np.float32).flatten()
            out = np.zeros(self.NUM_ACTIONS, dtype=np.float32)
            n = min(out.size, vec.size)
            if n > 0:
                out[:n] = np.maximum(vec[:n], 0.0)
            s = float(out.sum())
            if s > self.EPS:
                out /= s
            self._expected_next_move = out

    def set_opponent_policy_callback(
        self,
        callback: Optional[
            Callable[[Dict[str, Any], np.ndarray, np.random.Generator], Any]
        ],
    ) -> None:
        self.opponent_policy_callback = callback

    def set_opponent_profile(self, profile: Any, name: Optional[str] = None) -> None:
        self.opponent_profile = profile
        if name is not None:
            self.opponent_profile_name = str(name)

    def set_profile(self, profile: Any) -> None:
        self.set_opponent_profile(profile)

    def configure_opponent(
        self,
        profile_name: Optional[str] = None,
        profile: Any = None,
        callback: Optional[
            Callable[[Dict[str, Any], np.ndarray, np.random.Generator], Any]
        ] = None,
    ) -> None:
        if profile_name is not None:
            self.opponent_profile_name = str(profile_name)
        if profile is not None:
            self.opponent_profile = profile
        if callback is not None:
            self.set_opponent_policy_callback(callback)

    def clear_external_features(self) -> None:
        """Reset external observation placeholders to zeros."""
        self._opponent_intel = np.zeros(self.opponent_intel_size, dtype=np.float32)
        self._expected_next_move = np.zeros(self.NUM_ACTIONS, dtype=np.float32)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Gymnasium seeding: self.np_random becomes deterministic when seed is set.
        super().reset(seed=seed)
        options = options or {}

        if "opponent_intel" in options or "expected_next_move" in options:
            self.set_external_features(
                opponent_intel=options.get("opponent_intel"),
                expected_next_move=options.get("expected_next_move"),
            )

        self.hand_id += 1
        self._start_new_hand(options=options)
        self._advance_until_agent_turn_or_terminal()

        return self._get_obs(), self._build_info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.hand_over:
            return self._get_obs(), 0.0, True, False, self._build_info()

        if self.acting_player != self.agent_seat:
            self._advance_until_agent_turn_or_terminal()

        if self.hand_over:
            reward = float(self.chip_delta / self.big_blind)
            return self._get_obs(), reward, True, False, self._build_info()

        if self.acting_player != self.agent_seat:
            raise RuntimeError("step() was called while it is not the agent's turn")

        resolved_action, was_illegal = self._apply_action(self.agent_seat, int(action))
        self._advance_until_agent_turn_or_terminal()

        reward = -self.illegal_action_penalty if was_illegal else 0.0
        if self.hand_over:
            reward += float(self.chip_delta / self.big_blind)

        info = self._build_info(
            original_action=int(action),
            resolved_action=int(resolved_action),
            illegal_action=bool(was_illegal),
        )
        if was_illegal:
            info["illegal_action_penalty"] = -self.illegal_action_penalty

        return self._get_obs(), float(reward), self.hand_over, False, info

    def render(self) -> None:
        board = " ".join(self._card_to_str(c) for c in self.board) or "-"
        agent_hole = " ".join(self._card_to_str(c) for c in self.hole_cards[self.agent_seat])
        opp_hole = " ".join(self._card_to_str(c) for c in self.hole_cards[self.opponent_seat])
        print(
            f"hand={self.hand_id} street={self.STREET_NAMES[self.street]} "
            f"button={self.button} acting={self.acting_player} "
            f"pot={self.pot:.2f} stacks={self.stacks.round(2).tolist()} "
            f"to_call={self._to_call(self.acting_player):.2f} min_raise={self.min_raise:.2f}\n"
            f"board=[{board}] agent_hole=[{agent_hole}] opp_hole=[{opp_hole}] "
            f"hand_over={self.hand_over} showdown={self.showdown} winner={self.winner}"
        )

    # -------------------------------------------------------------------------
    # Core game flow
    # -------------------------------------------------------------------------
    def _init_hand_state(self) -> None:
        self.deck: list[int] = []
        self.hole_cards: list[list[int]] = [[], []]
        self.board: list[int] = []

        self.stacks = np.full(2, self.starting_stack, dtype=np.float64)
        self.street_contrib = np.zeros(2, dtype=np.float64)
        self.pot = 0.0

        self.street = self.STREET_PREFLOP
        self.button = self.agent_seat
        self.acting_player = self.agent_seat
        self.players_to_act: set[int] = set()

        self.current_bet = 0.0
        self.last_raise_size = self.big_blind
        self.min_raise = self.big_blind

        self.hand_over = False
        self.showdown = False
        self.winner: Optional[int] = None
        self.won_hand = False
        self.chip_delta = 0.0

    def _start_new_hand(self, *, options: Dict[str, Any]) -> None:
        self._init_hand_state()
        self.stacks[:] = self.starting_stack
        self.decision_index = 0
        self.decision_timestamp_s = 0.0

        if "button" in options:
            button = int(options["button"])
            if button not in (0, 1):
                raise ValueError("options['button'] must be 0 or 1")
            self.button = button
        else:
            self.button = int(self.np_random.integers(0, 2))

        sb_player = self.button
        bb_player = 1 - sb_player

        self.deck = list(range(52))
        self.np_random.shuffle(self.deck)

        # Deal private cards round-robin from the button.
        for _ in range(2):
            self.hole_cards[sb_player].append(self.deck.pop())
            self.hole_cards[bb_player].append(self.deck.pop())

        # Post blinds.
        self._commit_blind(sb_player, self.small_blind)
        self._commit_blind(bb_player, self.big_blind)

        self.current_bet = float(np.max(self.street_contrib))
        self.last_raise_size = self.big_blind
        self.min_raise = self.last_raise_size

        # Preflop: button (small blind) acts first in heads-up.
        self.acting_player = sb_player
        self.players_to_act = {sb_player, bb_player}

        # If stacks are tiny and a player is already all-in from blinds, runout.
        if (self.stacks[0] <= self.EPS or self.stacks[1] <= self.EPS) and self._to_call(
            self.acting_player
        ) <= self.EPS:
            self._runout_and_showdown()

    def _advance_until_agent_turn_or_terminal(self) -> None:
        safety = 0
        while not self.hand_over and self.acting_player != self.agent_seat:
            safety += 1
            if safety > 256:
                raise RuntimeError("loop safety triggered while advancing opponent actions")
            opp_action = self._choose_opponent_action(self.acting_player)
            self._apply_action(self.acting_player, opp_action)

    def _apply_action(self, player: int, requested_action: int) -> Tuple[int, bool]:
        if self.hand_over:
            return requested_action, False
        if player != self.acting_player:
            raise RuntimeError("attempted to act out of turn")

        legal_mask = self._legal_action_mask(player)
        resolved_action, was_illegal = self._map_action_to_legal(
            requested_action, legal_mask, player
        )
        self._advance_decision_clock()

        to_call = self._to_call(player)
        self.players_to_act.discard(player)

        if resolved_action == self.ACTION_FOLD:
            self._finish_hand(winner=1 - player, showdown=False)
            return resolved_action, was_illegal

        if resolved_action == self.ACTION_CHECK:
            if self._round_complete():
                self._complete_betting_round()
            else:
                self._switch_turn()
            return resolved_action, was_illegal

        if resolved_action == self.ACTION_CALL:
            call_amount = min(self.stacks[player], to_call)
            self._commit_chips(player, call_amount)

            if self._round_complete():
                self._complete_betting_round()
            else:
                self._switch_turn()
            return resolved_action, was_illegal

        # Aggressive actions: bet/raise/all-in.
        target_total = self._target_total_after_aggression(player, resolved_action)
        current_total = self.street_contrib[player]
        added = min(self.stacks[player], max(0.0, target_total - current_total))
        self._commit_chips(player, added)
        new_total = float(self.street_contrib[player])

        if new_total <= self.current_bet + self.EPS:
            # Could not exceed current bet (short all-in / effectively call).
            if self._round_complete():
                self._complete_betting_round()
            else:
                self._switch_turn()
            return resolved_action, was_illegal

        raise_size = new_total - self.current_bet
        self.current_bet = new_total

        # Full raise updates min-raise threshold; short all-in does not.
        if raise_size + self.EPS >= self.last_raise_size:
            self.last_raise_size = raise_size
            self.min_raise = self.last_raise_size

        self.players_to_act = {1 - player}
        self._switch_turn()
        return resolved_action, was_illegal

    def _round_complete(self) -> bool:
        if self.hand_over:
            return False
        if self.players_to_act:
            return False
        return abs(self.street_contrib[0] - self.street_contrib[1]) <= self.EPS

    def _complete_betting_round(self) -> None:
        if self.hand_over:
            return

        # If somebody is all-in and no one owes chips, run out board.
        if (
            (self.stacks[0] <= self.EPS or self.stacks[1] <= self.EPS)
            and self._to_call(0) <= self.EPS
            and self._to_call(1) <= self.EPS
        ):
            self._runout_and_showdown()
            return

        if self.street == self.STREET_RIVER:
            self._resolve_showdown()
            return

        self._deal_next_street()
        self.street_contrib[:] = 0.0
        self.current_bet = 0.0
        self.last_raise_size = self.big_blind
        self.min_raise = self.big_blind

        # Postflop in heads-up: non-button acts first.
        self.acting_player = 1 - self.button
        self.players_to_act = {self.acting_player, 1 - self.acting_player}

        if self.stacks[0] <= self.EPS or self.stacks[1] <= self.EPS:
            self._runout_and_showdown()

    def _deal_next_street(self) -> None:
        # Burn card before each community dealing street.
        self._burn_one()
        if self.street == self.STREET_PREFLOP:
            self.street = self.STREET_FLOP
            for _ in range(3):
                self.board.append(self.deck.pop())
        elif self.street == self.STREET_FLOP:
            self.street = self.STREET_TURN
            self.board.append(self.deck.pop())
        elif self.street == self.STREET_TURN:
            self.street = self.STREET_RIVER
            self.board.append(self.deck.pop())
        else:
            raise RuntimeError("cannot deal beyond river")

    def _runout_and_showdown(self) -> None:
        while self.street < self.STREET_RIVER:
            self._deal_next_street()
        self._resolve_showdown()

    def _resolve_showdown(self) -> None:
        score_0 = self._best_hand_rank(self.hole_cards[0] + self.board)
        score_1 = self._best_hand_rank(self.hole_cards[1] + self.board)

        if score_0 > score_1:
            winner: Optional[int] = 0
        elif score_1 > score_0:
            winner = 1
        else:
            winner = None

        self._finish_hand(winner=winner, showdown=True)

    def _finish_hand(self, *, winner: Optional[int], showdown: bool) -> None:
        self.hand_over = True
        self.showdown = bool(showdown)
        self.winner = winner

        if winner is None:
            split = self.pot * 0.5
            self.stacks[0] += split
            self.stacks[1] += split
        else:
            self.stacks[winner] += self.pot

        self.players_to_act.clear()
        self.chip_delta = float(self.stacks[self.agent_seat] - self.starting_stack)
        self.won_hand = self.chip_delta > self.EPS

    def _switch_turn(self) -> None:
        self.acting_player = 1 - self.acting_player

    # -------------------------------------------------------------------------
    # Betting helpers
    # -------------------------------------------------------------------------
    def _commit_blind(self, player: int, amount: float) -> None:
        self._commit_chips(player, min(amount, self.stacks[player]))

    def _commit_chips(self, player: int, amount: float) -> None:
        if amount <= self.EPS:
            return
        amount = float(min(amount, self.stacks[player]))
        self.stacks[player] -= amount
        self.street_contrib[player] += amount
        self.pot += amount

    def _to_call(self, player: int) -> float:
        return float(max(0.0, self.current_bet - self.street_contrib[player]))

    def _legal_action_mask(self, player: int) -> np.ndarray:
        mask = np.zeros(self.NUM_ACTIONS, dtype=np.float32)
        stack = float(self.stacks[player])
        opp_stack = float(self.stacks[1 - player])
        to_call = self._to_call(player)
        can_aggress = stack > self.EPS and opp_stack > self.EPS

        if to_call > self.EPS:
            mask[self.ACTION_FOLD] = 1.0
            mask[self.ACTION_CALL] = 1.0

            if can_aggress and (stack - to_call) > self.EPS:
                max_total = self.street_contrib[player] + stack
                min_total = self.current_bet + self.min_raise
                if max_total + self.EPS >= min_total:
                    mask[self.ACTION_RAISE_2X_POT] = 1.0
        else:
            mask[self.ACTION_CHECK] = 1.0
            if can_aggress:
                mask[self.ACTION_BET_HALF_POT] = 1.0
                mask[self.ACTION_BET_POT] = 1.0

        if stack > self.EPS:
            mask[self.ACTION_ALL_IN] = 1.0

        return mask

    def _map_action_to_legal(
        self, action: int, legal_mask: np.ndarray, player: int
    ) -> Tuple[int, bool]:
        if 0 <= action < self.NUM_ACTIONS and legal_mask[action] > 0.5:
            return int(action), False

        to_call = self._to_call(player)
        fallback: list[int]

        if to_call <= self.EPS:
            if action in (self.ACTION_FOLD, self.ACTION_CALL):
                fallback = [
                    self.ACTION_CHECK,
                    self.ACTION_BET_HALF_POT,
                    self.ACTION_BET_POT,
                    self.ACTION_ALL_IN,
                ]
            elif action == self.ACTION_RAISE_2X_POT:
                fallback = [
                    self.ACTION_BET_POT,
                    self.ACTION_BET_HALF_POT,
                    self.ACTION_ALL_IN,
                    self.ACTION_CHECK,
                ]
            else:
                fallback = [self.ACTION_CHECK, self.ACTION_BET_HALF_POT, self.ACTION_BET_POT]
        else:
            if action == self.ACTION_CHECK:
                fallback = [self.ACTION_CALL, self.ACTION_FOLD, self.ACTION_ALL_IN]
            elif action in (self.ACTION_BET_HALF_POT, self.ACTION_BET_POT):
                fallback = [
                    self.ACTION_RAISE_2X_POT,
                    self.ACTION_CALL,
                    self.ACTION_ALL_IN,
                    self.ACTION_FOLD,
                ]
            else:
                fallback = [self.ACTION_CALL, self.ACTION_FOLD, self.ACTION_ALL_IN]

        for candidate in fallback:
            if legal_mask[candidate] > 0.5:
                return candidate, True

        legal = np.flatnonzero(legal_mask > 0.5)
        if legal.size > 0:
            return int(legal[0]), True
        return self.ACTION_CHECK, True

    def _target_total_after_aggression(self, player: int, action: int) -> float:
        current_total = float(self.street_contrib[player])
        max_total = current_total + float(self.stacks[player])
        to_call = self._to_call(player)

        if action == self.ACTION_ALL_IN:
            return max_total

        if action == self.ACTION_BET_HALF_POT:
            # Opening bet: half of current pot, at least 1 BB.
            target = current_total + max(self.big_blind, 0.5 * self.pot)
        elif action == self.ACTION_BET_POT:
            # Opening bet: pot-sized, at least 1 BB.
            target = current_total + max(self.big_blind, self.pot)
        elif action == self.ACTION_RAISE_2X_POT:
            # Raise uses a coarse overbet abstraction: +2x current pot.
            increment = max(self.min_raise, 2.0 * self.pot)
            target = self.current_bet + increment
        else:
            target = current_total + to_call

        if to_call > self.EPS:
            # True raises must at least satisfy min-raise unless all-in short.
            target = max(target, self.current_bet + self.min_raise)
        else:
            target = max(target, current_total + self.big_blind)

        target = min(target, max_total)
        if target <= self.current_bet + self.EPS and max_total > self.current_bet + self.EPS:
            target = max_total
        return float(target)

    # -------------------------------------------------------------------------
    # Opponent policy (simple heuristic baseline)
    # -------------------------------------------------------------------------
    def _choose_opponent_action(self, player: int) -> int:
        mask = self._legal_action_mask(player)
        callback_action = self._choose_opponent_action_from_callback(player, mask)
        if callback_action is not None:
            return callback_action

        to_call = self._to_call(player)
        strength = self._hole_strength_proxy(player)
        rng = self.np_random

        if to_call > self.EPS:
            pot_odds = to_call / max(self.EPS, self.pot + to_call)

            if mask[self.ACTION_FOLD] > 0.5 and strength + 0.08 < pot_odds:
                if rng.random() < 0.85:
                    return self.ACTION_FOLD

            if mask[self.ACTION_RAISE_2X_POT] > 0.5 and strength > 0.78:
                if rng.random() < 0.35:
                    return self.ACTION_RAISE_2X_POT

            if mask[self.ACTION_ALL_IN] > 0.5 and strength > 0.92:
                if rng.random() < 0.22:
                    return self.ACTION_ALL_IN

            if mask[self.ACTION_CALL] > 0.5:
                return self.ACTION_CALL
            if mask[self.ACTION_ALL_IN] > 0.5:
                return self.ACTION_ALL_IN
            return self.ACTION_FOLD

        # to_call == 0
        if mask[self.ACTION_CHECK] <= 0.5:
            legal = np.flatnonzero(mask > 0.5)
            return int(legal[0]) if legal.size else self.ACTION_CHECK

        aggression_score = strength + 0.15 * float(rng.random())
        if mask[self.ACTION_ALL_IN] > 0.5 and aggression_score > 0.96 and self.stacks[player] <= 20 * self.big_blind:
            return self.ACTION_ALL_IN
        if mask[self.ACTION_BET_POT] > 0.5 and aggression_score > 0.76:
            return self.ACTION_BET_POT
        if mask[self.ACTION_BET_HALF_POT] > 0.5 and aggression_score > 0.57:
            return self.ACTION_BET_HALF_POT
        return self.ACTION_CHECK

    def _choose_opponent_action_from_callback(
        self,
        player: int,
        legal_mask: np.ndarray,
    ) -> Optional[int]:
        if self.opponent_policy_callback is None:
            return None

        state = self._build_opponent_state(player)
        try:
            callback_result = self.opponent_policy_callback(
                state,
                legal_mask.copy(),
                self.np_random,
            )
        except Exception:
            return None

        if callback_result is None:
            return None

        if isinstance(callback_result, Mapping) and "distribution" in callback_result:
            raw_distribution = callback_result.get("distribution", {})
            maybe_intel = callback_result.get("opponent_intel")
        else:
            raw_distribution = callback_result
            maybe_intel = None

        if not isinstance(raw_distribution, Mapping):
            return None

        probs = np.zeros(self.NUM_ACTIONS, dtype=np.float32)
        for raw_key, raw_value in raw_distribution.items():
            try:
                prob = float(raw_value)
            except (TypeError, ValueError):
                continue
            if prob < 0.0:
                continue

            if isinstance(raw_key, str):
                action_id = self.ACTION_NAME_TO_ID.get(raw_key.lower())
            else:
                try:
                    action_id = int(raw_key)
                except (TypeError, ValueError):
                    action_id = None

            if action_id is None or not (0 <= action_id < self.NUM_ACTIONS):
                continue
            probs[action_id] += prob

        probs_sum = float(probs.sum())
        if probs_sum <= self.EPS:
            return None
        probs /= probs_sum

        # Store the latest expected move distribution in observation placeholders.
        self.set_external_features(expected_next_move=probs)
        if maybe_intel is not None:
            self.set_external_features(opponent_intel=maybe_intel)

        legal_probs = probs * legal_mask
        legal_sum = float(legal_probs.sum())
        if legal_sum <= self.EPS:
            return None
        legal_probs /= legal_sum

        action = int(self.np_random.choice(np.arange(self.NUM_ACTIONS), p=legal_probs))
        return action

    # -------------------------------------------------------------------------
    # Observation / info
    # -------------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        street_one_hot = np.zeros(4, dtype=np.float32)
        street_one_hot[self.street] = 1.0

        to_call = self._to_call(self.agent_seat) if not self.hand_over else 0.0
        legal_mask = (
            self._legal_action_mask(self.agent_seat)
            if (not self.hand_over and self.acting_player == self.agent_seat)
            else np.zeros(self.NUM_ACTIONS, dtype=np.float32)
        )

        self_stack_norm = self._norm_stack(self.stacks[self.agent_seat])
        opp_stack_norm = self._norm_stack(self.stacks[self.opponent_seat])
        pot_norm = np.clip(self.pot / (2.0 * self.starting_stack), 0.0, 1.0)
        to_call_norm = self._norm_stack(to_call)
        min_raise_norm = self._norm_stack(self.min_raise)
        button_indicator = 1.0 if self.button == self.agent_seat else 0.0

        hole_strength = np.array([self._hole_strength_proxy(self.agent_seat)], dtype=np.float32)
        board_texture = self._board_texture_features()

        obs = np.concatenate(
            [
                street_one_hot,
                np.array(
                    [
                        self_stack_norm,
                        opp_stack_norm,
                        pot_norm,
                        to_call_norm,
                        min_raise_norm,
                    ],
                    dtype=np.float32,
                ),
                np.array([button_indicator], dtype=np.float32),
                legal_mask.astype(np.float32),
                hole_strength,
                board_texture,
                self._opponent_intel.astype(np.float32),
                self._expected_next_move.astype(np.float32),
            ]
        ).astype(np.float32)
        return obs

    def _build_info(
        self,
        *,
        original_action: Optional[int] = None,
        resolved_action: Optional[int] = None,
        illegal_action: bool = False,
    ) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "chip_delta": float(self.chip_delta if self.hand_over else 0.0),
            "chip_delta_bb": float(self.chip_delta / self.big_blind if self.hand_over else 0.0),
            "won_hand": bool(self.won_hand if self.hand_over else False),
            "showdown": bool(self.showdown if self.hand_over else False),
            "pot": float(self.pot),
            "pot_bb": float(self.pot / self.big_blind),
            "hand_id": int(self.hand_id),
            "street": self.STREET_NAMES[self.street],
            "decision_timestamp_s": float(self.decision_timestamp_s),
        }
        if original_action is not None:
            info["original_action"] = int(original_action)
        if resolved_action is not None:
            info["resolved_action"] = int(resolved_action)
            info["resolved_action_name"] = self.ACTION_NAMES[int(resolved_action)]
        if original_action is not None and resolved_action is not None:
            info["illegal_action_mapped"] = bool(illegal_action)
        if self.hand_over:
            info["winner"] = self.winner
        return info

    def _norm_stack(self, value: float) -> float:
        return float(np.clip(value / self.starting_stack, 0.0, 1.0))

    def _advance_decision_clock(self) -> None:
        self.decision_index += 1
        self.decision_timestamp_s = float(self.decision_index * self.decision_interval_s)

    def _build_opponent_state(self, player: int) -> Dict[str, Any]:
        board_texture = self._board_texture_features()
        board_danger = float(
            np.clip(0.45 * board_texture[1] + 0.35 * board_texture[2] + 0.20 * board_texture[3], 0.0, 1.0)
        )
        first_actor = self._first_to_act_this_street()
        position_advantage = 1.0 if player != first_actor else 0.0
        state = {
            "hand_id": int(self.hand_id),
            "street": self.STREET_NAMES[self.street],
            "pot_size": float(self.pot),
            "to_call": float(self._to_call(player)),
            "stack": float(self.stacks[player]),
            "opponent_stack": float(self.stacks[1 - player]),
            "hand_strength": float(self._hole_strength_proxy(player)),
            "position_advantage": float(position_advantage),
            "board_danger": board_danger,
            "facing_bet": bool(self._to_call(player) > self.EPS),
            "timestamp_s": float(self.decision_timestamp_s),
        }
        return state

    def _first_to_act_this_street(self) -> int:
        if self.street == self.STREET_PREFLOP:
            return self.button
        return 1 - self.button

    # -------------------------------------------------------------------------
    # Feature engineering helpers
    # -------------------------------------------------------------------------
    def _hole_strength_proxy(self, player: int) -> float:
        hole = self.hole_cards[player]
        ranks = sorted((self._card_rank(c) for c in hole), reverse=True)
        suits = [self._card_suit(c) for c in hole]

        # Simple preflop strength heuristic.
        high_pair_bonus = 0.32 if ranks[0] == ranks[1] else 0.0
        high_cards = (ranks[0] + ranks[1]) / 28.0
        gap = abs(ranks[0] - ranks[1])
        connector_bonus = max(0.0, 0.12 - 0.03 * max(0, gap - 1))
        suited_bonus = 0.05 if suits[0] == suits[1] else 0.0
        ace_bonus = 0.03 if 14 in ranks else 0.0
        preflop_strength = np.clip(
            0.62 * high_cards + high_pair_bonus + connector_bonus + suited_bonus + ace_bonus,
            0.0,
            1.0,
        )

        if len(self.board) < 3:
            return float(preflop_strength)

        # Blend in made-hand and draw quality postflop.
        combo = hole + self.board
        made = self._best_hand_rank(combo)
        category_norm = made[0] / 8.0
        tie_norm = (made[1] / 14.0) if len(made) > 1 else 0.0
        draw_bonus = self._draw_potential(player)

        strength = np.clip(
            0.5 * preflop_strength + 0.45 * category_norm + 0.05 * tie_norm + draw_bonus,
            0.0,
            1.0,
        )
        return float(strength)

    def _draw_potential(self, player: int) -> float:
        if len(self.board) < 3 or len(self.board) >= 5:
            return 0.0
        cards = self.hole_cards[player] + self.board
        suit_counts = Counter(self._card_suit(c) for c in cards)
        flush_draw = 1.0 if max(suit_counts.values()) >= 4 else 0.0
        straight_run = self._longest_consecutive_run([self._card_rank(c) for c in cards])
        straight_draw = 1.0 if straight_run >= 4 else 0.0
        return 0.06 * flush_draw + 0.05 * straight_draw

    def _board_texture_features(self) -> np.ndarray:
        if not self.board:
            return np.zeros(5, dtype=np.float32)

        ranks = [self._card_rank(c) for c in self.board]
        suits = [self._card_suit(c) for c in self.board]
        rank_counts = Counter(ranks)
        suit_counts = Counter(suits)

        board_size_norm = len(self.board) / 5.0
        suitedness = max(suit_counts.values()) / 5.0
        pairedness = 1.0 if max(rank_counts.values()) >= 2 else 0.0
        straightiness = self._longest_consecutive_run(ranks) / 5.0
        high_card_density = sum(1 for r in ranks if r >= 11) / max(1, len(ranks))

        return np.array(
            [board_size_norm, suitedness, pairedness, straightiness, high_card_density],
            dtype=np.float32,
        )

    # -------------------------------------------------------------------------
    # Hand evaluation (showdown)
    # -------------------------------------------------------------------------
    def _best_hand_rank(self, cards: Sequence[int]) -> Tuple[int, ...]:
        if len(cards) < 5:
            raise ValueError("need at least 5 cards to evaluate hand rank")
        best: Optional[Tuple[int, ...]] = None
        for combo in itertools.combinations(cards, 5):
            score = self._score_five_card_hand(combo)
            if best is None or score > best:
                best = score
        if best is None:
            raise RuntimeError("failed to evaluate hand")
        return best

    def _score_five_card_hand(self, cards: Sequence[int]) -> Tuple[int, ...]:
        ranks = sorted((self._card_rank(c) for c in cards), reverse=True)
        suits = [self._card_suit(c) for c in cards]
        counts = Counter(ranks)
        by_count = sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)

        is_flush = len(set(suits)) == 1
        straight_high = self._straight_high_from_ranks(ranks)

        if is_flush and straight_high > 0:
            return (8, straight_high)  # straight flush

        if by_count[0][1] == 4:
            four = by_count[0][0]
            kicker = max(r for r in ranks if r != four)
            return (7, four, kicker)

        if by_count[0][1] == 3 and by_count[1][1] == 2:
            return (6, by_count[0][0], by_count[1][0])  # full house

        if is_flush:
            return (5, *sorted(ranks, reverse=True))

        if straight_high > 0:
            return (4, straight_high)

        if by_count[0][1] == 3:
            trips = by_count[0][0]
            kickers = sorted([r for r in ranks if r != trips], reverse=True)
            return (3, trips, *kickers)

        if by_count[0][1] == 2 and by_count[1][1] == 2:
            pair_hi = max(by_count[0][0], by_count[1][0])
            pair_lo = min(by_count[0][0], by_count[1][0])
            kicker = max(r for r in ranks if r not in (pair_hi, pair_lo))
            return (2, pair_hi, pair_lo, kicker)

        if by_count[0][1] == 2:
            pair = by_count[0][0]
            kickers = sorted([r for r in ranks if r != pair], reverse=True)
            return (1, pair, *kickers)

        return (0, *sorted(ranks, reverse=True))  # high card

    def _straight_high_from_ranks(self, ranks: Sequence[int]) -> int:
        uniq = sorted(set(ranks))
        if 14 in uniq:
            uniq = [1] + uniq
        run = 1
        best = 0
        for i in range(1, len(uniq)):
            if uniq[i] == uniq[i - 1] + 1:
                run += 1
                if run >= 5:
                    best = uniq[i]
            else:
                run = 1
        return int(best)

    def _longest_consecutive_run(self, ranks: Sequence[int]) -> int:
        if not ranks:
            return 0
        uniq = sorted(set(ranks))
        if 14 in uniq:
            uniq = [1] + uniq
        run = 1
        best = 1
        for i in range(1, len(uniq)):
            if uniq[i] == uniq[i - 1] + 1:
                run += 1
                best = max(best, run)
            else:
                run = 1
        return int(best)

    # -------------------------------------------------------------------------
    # Card helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _card_rank(card: int) -> int:
        # 2..14 (Ace high)
        return (card % 13) + 2

    @staticmethod
    def _card_suit(card: int) -> int:
        # 0..3
        return card // 13

    def _burn_one(self) -> None:
        if not self.deck:
            raise RuntimeError("cannot burn card from empty deck")
        self.deck.pop()

    def _card_to_str(self, card: int) -> str:
        ranks = "23456789TJQKA"
        suits = "cdhs"
        return f"{ranks[self._card_rank(card) - 2]}{suits[self._card_suit(card)]}"


# Backward-friendly alias.
HeadsUpNoLimitHoldemEnv = HoldemEnv

#!/usr/bin/env python3
"""Minimal adaptive chess model with mock facial tells.

What this script does:
1. Trains an adaptive opponent model from simulated games.
2. Uses player profiles + mock video tells (every 10s) to predict next opponent move.
3. At each game state during inference, prints the shortest mating line under the
   modeled opponent move distribution.

This is intentionally simple and lightweight.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import chess
import numpy as np

PIECE_VALUES: Dict[int, float] = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.2,
    chess.BISHOP: 3.3,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}

MOVE_LABELS: Tuple[str, ...] = (
    "capture_check",
    "capture",
    "check",
    "castle",
    "develop",
    "center",
    "quiet",
)

CENTER_SQUARES = {chess.D4, chess.E4, chess.D5, chess.E5}
EXTENDED_CENTER = {
    chess.C3,
    chess.D3,
    chess.E3,
    chess.F3,
    chess.C4,
    chess.F4,
    chess.C5,
    chess.F5,
    chess.C6,
    chess.D6,
    chess.E6,
    chess.F6,
}

DEFAULT_LOG_DIR = "/Users/sankalp/Projects/experiment/games/chess/logs"


@dataclass(frozen=True)
class PlayerProfile:
    name: str
    aggression: float
    tacticality: float
    solidness: float
    speed: float
    composure: float


@dataclass(frozen=True)
class VideoTellSnapshot:
    timestamp_s: float
    gaze_aversion: float
    blink_rate: float
    jaw_tension: float
    voice_stress: float
    confidence: float


DEFAULT_PROFILES: Dict[str, PlayerProfile] = {
    "balanced": PlayerProfile(
        name="balanced",
        aggression=0.50,
        tacticality=0.50,
        solidness=0.50,
        speed=0.55,
        composure=0.65,
    ),
    "tactical": PlayerProfile(
        name="tactical",
        aggression=0.70,
        tacticality=0.85,
        solidness=0.35,
        speed=0.65,
        composure=0.55,
    ),
    "solid": PlayerProfile(
        name="solid",
        aggression=0.30,
        tacticality=0.40,
        solidness=0.85,
        speed=0.40,
        composure=0.80,
    ),
    "blitzer": PlayerProfile(
        name="blitzer",
        aggression=0.75,
        tacticality=0.60,
        solidness=0.30,
        speed=0.90,
        composure=0.35,
    ),
    "endgame": PlayerProfile(
        name="endgame",
        aggression=0.35,
        tacticality=0.55,
        solidness=0.75,
        speed=0.35,
        composure=0.85,
    ),
}


@dataclass(frozen=True)
class RunArtifacts:
    run_id: str
    run_dir: Path
    run_meta: Path
    training_games: Path
    inference_trace: Path
    run_summary: Path
    model_snapshot: Path


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def build_event(event_type: str, **payload: object) -> Dict[str, object]:
    event: Dict[str, object] = {
        "timestamp": utc_timestamp(),
        "event_type": event_type,
    }
    event.update(payload)
    return event


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True))
        f.write("\n")


def init_run_artifacts(log_dir: str, run_id: Optional[str]) -> RunArtifacts:
    resolved_run_id = run_id.strip() if run_id and run_id.strip() else ""
    if not resolved_run_id:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        resolved_run_id = f"run_{stamp}_{os.getpid()}"

    run_dir = Path(log_dir).expanduser().resolve() / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    artifacts = RunArtifacts(
        run_id=resolved_run_id,
        run_dir=run_dir,
        run_meta=run_dir / "run_meta.json",
        training_games=run_dir / "training_games.jsonl",
        inference_trace=run_dir / "inference_trace.jsonl",
        run_summary=run_dir / "run_summary.json",
        model_snapshot=run_dir / "model_snapshot.json",
    )

    # Truncate JSONL artifacts for deterministic reruns with the same run_id.
    artifacts.training_games.write_text("", encoding="utf-8")
    artifacts.inference_trace.write_text("", encoding="utf-8")
    return artifacts


def clamp01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


class MockVideoFeed:
    """Deterministic mock tells sampled every `interval_s` seconds."""

    def __init__(self, profile: PlayerProfile, seed: int = 0, interval_s: int = 10) -> None:
        self.profile = profile
        self.seed = int(seed)
        self.interval_s = int(interval_s)

    def snapshot_at(self, timestamp_s: float) -> VideoTellSnapshot:
        index = int(max(0.0, timestamp_s) // self.interval_s)
        base_seed = (
            self.seed
            + index * 997
            + int(self.profile.aggression * 1000)
            + int(self.profile.tacticality * 2000)
            + int(self.profile.solidness * 3000)
        )
        rng = np.random.default_rng(base_seed)

        fatigue = clamp01(index / 70.0)
        volatility = clamp01(1.0 - self.profile.composure + 0.3 * self.profile.speed)

        gaze_aversion = clamp01(
            0.18
            + 0.35 * volatility
            + 0.12 * fatigue
            + rng.normal(0.0, 0.06)
        )
        blink_rate = clamp01(
            0.22
            + 0.30 * volatility
            + 0.18 * self.profile.speed
            + 0.15 * fatigue
            + rng.normal(0.0, 0.07)
        )
        jaw_tension = clamp01(
            0.20
            + 0.30 * volatility
            + 0.20 * self.profile.aggression
            + rng.normal(0.0, 0.06)
        )
        voice_stress = clamp01(
            0.15
            + 0.32 * volatility
            + 0.16 * fatigue
            + rng.normal(0.0, 0.06)
        )
        confidence = clamp01(
            0.30
            + 0.25 * self.profile.composure
            + 0.20 * self.profile.tacticality
            + 0.10 * self.profile.solidness
            - 0.20 * voice_stress
            + rng.normal(0.0, 0.05)
        )

        return VideoTellSnapshot(
            timestamp_s=float(index * self.interval_s),
            gaze_aversion=gaze_aversion,
            blink_rate=blink_rate,
            jaw_tension=jaw_tension,
            voice_stress=voice_stress,
            confidence=confidence,
        )


class AdaptiveMoveModel:
    """Simple adaptive model over coarse move labels with contextual counts."""

    def __init__(self) -> None:
        # Dirichlet-style counts for each context, initialized with 1 per label.
        self.context_counts: Dict[str, Counter[str]] = defaultdict(
            lambda: Counter({label: 1.0 for label in MOVE_LABELS})
        )
        # Exact move counts per context, keyed by move UCI.
        self.context_move_counts: Dict[str, Counter[str]] = defaultdict(Counter)
        self.label_smoothing = 0.5
        self.move_smoothing = 1.0

    def _phase(self, board: chess.Board) -> str:
        pieces = len(board.piece_map())
        if board.fullmove_number <= 10:
            return "opening"
        if pieces <= 10:
            return "endgame"
        return "middlegame"

    def _material_balance(self, board: chess.Board, color: chess.Color) -> float:
        mine = 0.0
        theirs = 0.0
        for sq, piece in board.piece_map().items():
            val = PIECE_VALUES[piece.piece_type]
            if piece.color == color:
                mine += val
            else:
                theirs += val
        return mine - theirs

    def _tell_bucket(self, tell: VideoTellSnapshot) -> str:
        stress = clamp01(0.45 * tell.voice_stress + 0.30 * tell.jaw_tension + 0.25 * tell.blink_rate)
        conf = tell.confidence
        if stress > 0.65:
            return "stressed"
        if conf > 0.65:
            return "confident"
        return "neutral"

    def context_key(self, board: chess.Board, color: chess.Color, tell: VideoTellSnapshot) -> str:
        mat = self._material_balance(board, color)
        if mat > 1.5:
            mat_bucket = "ahead"
        elif mat < -1.5:
            mat_bucket = "behind"
        else:
            mat_bucket = "equal"
        check_tag = "in_check" if board.is_check() else "safe"
        return f"{self._phase(board)}|{mat_bucket}|{check_tag}|{self._tell_bucket(tell)}"

    def move_label(self, board: chess.Board, move: chess.Move) -> str:
        is_capture = board.is_capture(move)
        gives_check = board.gives_check(move)
        if is_capture and gives_check:
            return "capture_check"
        if is_capture:
            return "capture"
        if gives_check:
            return "check"
        if board.is_castling(move):
            return "castle"

        from_sq = move.from_square
        to_sq = move.to_square
        piece = board.piece_at(from_sq)
        if piece and piece.piece_type in (chess.KNIGHT, chess.BISHOP):
            from_rank = chess.square_rank(from_sq)
            home_rank = 0 if piece.color == chess.WHITE else 7
            if from_rank == home_rank:
                return "develop"

        if to_sq in CENTER_SQUARES or to_sq in EXTENDED_CENTER:
            return "center"
        return "quiet"

    def _label_biases(self, profile: PlayerProfile, tell: VideoTellSnapshot) -> Dict[str, float]:
        stress = clamp01(0.45 * tell.voice_stress + 0.30 * tell.jaw_tension + 0.25 * tell.blink_rate)
        conf = tell.confidence

        aggr = clamp01(profile.aggression * (0.7 + 0.3 * conf))
        tact = clamp01(profile.tacticality * (0.75 + 0.25 * conf))
        solid = clamp01(profile.solidness * (0.75 + 0.25 * (1.0 - stress)))

        return {
            "capture_check": 1.5 * tact + 0.8 * aggr - 0.2 * stress,
            "capture": 1.0 * aggr + 0.5 * tact - 0.1 * stress,
            "check": 1.1 * tact + 0.6 * aggr - 0.1 * stress,
            "castle": 1.2 * solid + 0.3 * (1.0 - aggr),
            "develop": 0.8 * solid + 0.3 * profile.composure,
            "center": 0.5 * profile.tacticality + 0.4 * profile.solidness,
            "quiet": 0.7 * solid + 0.4 * (1.0 - aggr) + 0.4 * stress,
        }

    def predict_distribution(
        self,
        board: chess.Board,
        profile: PlayerProfile,
        tell: VideoTellSnapshot,
    ) -> Dict[chess.Move, float]:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return {}

        ctx = self.context_key(board, board.turn, tell)
        counts = self.context_counts[ctx]
        move_counts = self.context_move_counts[ctx]
        biases = self._label_biases(profile, tell)
        label_total = float(sum(counts.values()))
        label_denom = label_total + self.label_smoothing * len(MOVE_LABELS)
        move_total = float(sum(move_counts.values()))
        move_denom = move_total + self.move_smoothing * max(1, len(legal_moves))

        scores = []
        for move in legal_moves:
            label = self.move_label(board, move)
            label_component = (float(counts[label]) + self.label_smoothing) / max(label_denom, 1e-8)
            move_count = float(move_counts.get(move.uci(), 0.0))
            # Keep unseen moves neutral while boosting frequently seen moves.
            move_component = 1.0 + (move_count / max(move_denom, 1e-8))
            tactical_component = math.exp(biases.get(label, 0.0))
            score = label_component * move_component * tactical_component

            # Quick tactical priors.
            board.push(move)
            if board.is_checkmate():
                score *= 1000.0
            elif board.is_check():
                score *= 1.4
            board.pop()

            # Prefer captures that win material.
            if board.is_capture(move):
                captured_piece = board.piece_at(move.to_square)
                mover_piece = board.piece_at(move.from_square)
                if captured_piece is not None and mover_piece is not None:
                    trade_gain = PIECE_VALUES[captured_piece.piece_type] - PIECE_VALUES[mover_piece.piece_type] * 0.15
                    score *= 1.0 + max(0.0, trade_gain) * 0.12
            scores.append(max(score, 1e-8))

        probs = np.asarray(scores, dtype=np.float64)
        probs /= probs.sum()
        return {move: float(prob) for move, prob in zip(legal_moves, probs)}

    def update(self, board: chess.Board, played_move: chess.Move, tell: VideoTellSnapshot) -> None:
        ctx = self.context_key(board, board.turn, tell)
        label = self.move_label(board, played_move)
        self.context_counts[ctx][label] += 1.0
        self.context_move_counts[ctx][played_move.uci()] += 1.0


class TrueOpponentPolicy:
    """Hidden simulator policy used to generate training observations."""

    def choose_move(
        self,
        board: chess.Board,
        profile: PlayerProfile,
        tell: VideoTellSnapshot,
        rng: random.Random,
    ) -> chess.Move:
        moves = list(board.legal_moves)
        if not moves:
            raise RuntimeError("No legal moves")

        stress = clamp01(0.45 * tell.voice_stress + 0.30 * tell.jaw_tension + 0.25 * tell.blink_rate)
        confidence = tell.confidence

        scores: List[float] = []
        for move in moves:
            score = 1.0
            is_capture = board.is_capture(move)
            gives_check = board.gives_check(move)
            is_castle = board.is_castling(move)

            if is_capture:
                score += 1.5 * profile.aggression
            if gives_check:
                score += 1.8 * profile.tacticality
            if is_castle:
                score += 1.2 * profile.solidness
            if move.to_square in CENTER_SQUARES:
                score += 0.6 * profile.solidness

            board.push(move)
            if board.is_checkmate():
                board.pop()
                return move
            board.pop()

            # Material capture quality.
            if is_capture:
                captured = board.piece_at(move.to_square)
                mover = board.piece_at(move.from_square)
                if captured is not None and mover is not None:
                    gain = PIECE_VALUES[captured.piece_type] - PIECE_VALUES[mover.piece_type] * 0.2
                    score += 0.4 * max(0.0, gain)

            # Stress causes simplification / lower tactical sharpness.
            score *= 1.0 + 0.15 * confidence - 0.12 * stress

            # Blunder/noise term from speed+stress (simple stochasticity).
            noise_scale = 0.20 + 0.35 * profile.speed + 0.30 * stress - 0.20 * profile.composure
            score += rng.uniform(-noise_scale, noise_scale)
            scores.append(max(score, 1e-6))

        probs = np.asarray(scores, dtype=np.float64)
        probs /= probs.sum()
        idx = int(np.random.default_rng(rng.randint(1, 10_000_000)).choice(len(moves), p=probs))
        return moves[idx]


def choose_simple_agent_move(board: chess.Board, rng: random.Random) -> chess.Move:
    """Simple training-time white-side policy (non-adaptive)."""
    moves = list(board.legal_moves)
    if not moves:
        raise RuntimeError("No legal moves")

    for move in moves:
        board.push(move)
        is_mate = board.is_checkmate()
        board.pop()
        if is_mate:
            return move

    checks = [m for m in moves if board.gives_check(m)]
    if checks and rng.random() < 0.55:
        return rng.choice(checks)

    captures = [m for m in moves if board.is_capture(m)]
    if captures and rng.random() < 0.6:
        return rng.choice(captures)

    return rng.choice(moves)


def train_adaptive_model(
    model: AdaptiveMoveModel,
    profile: PlayerProfile,
    games: int,
    max_plies: int,
    seed: int,
    training_log_path: Optional[Path] = None,
) -> Dict[str, float]:
    rng = random.Random(seed)
    true_policy = TrueOpponentPolicy()

    observed_opponent_moves = 0
    game_lengths: List[int] = []
    prediction_samples = 0
    top1_hits = 0
    top3_hits = 0
    nll_sum = 0.0
    brier_sum = 0.0

    for g in range(games):
        board = chess.Board()
        feed = MockVideoFeed(profile=profile, seed=seed + g * 37, interval_s=10)

        ply = 0
        game_observed_opponent_moves = 0
        game_prediction_samples = 0
        game_top1_hits = 0
        game_top3_hits = 0
        game_nll_sum = 0.0
        game_brier_sum = 0.0

        while not board.is_game_over() and ply < max_plies:
            tell = feed.snapshot_at(ply * 10.0)
            if board.turn == chess.BLACK:
                move = true_policy.choose_move(board, profile, tell, rng)

                pred_dist = model.predict_distribution(board, profile, tell)
                ranked = sorted(pred_dist.items(), key=lambda x: x[1], reverse=True)
                top1_move = ranked[0][0] if ranked else None
                top3_moves = {mv for mv, _ in ranked[:3]}
                p_actual = max(float(pred_dist.get(move, 0.0)), 1e-12)

                if top1_move == move:
                    top1_hits += 1
                    game_top1_hits += 1
                if move in top3_moves:
                    top3_hits += 1
                    game_top3_hits += 1

                nll = -math.log(p_actual)
                nll_sum += nll
                game_nll_sum += nll

                legal_moves = list(board.legal_moves)
                brier = 0.0
                for legal_move in legal_moves:
                    p = float(pred_dist.get(legal_move, 0.0))
                    y = 1.0 if legal_move == move else 0.0
                    brier += (p - y) ** 2
                brier /= float(max(1, len(legal_moves)))
                brier_sum += brier
                game_brier_sum += brier

                prediction_samples += 1
                game_prediction_samples += 1

                model.update(board, move, tell)
                observed_opponent_moves += 1
                game_observed_opponent_moves += 1
            else:
                move = choose_simple_agent_move(board, rng)

            board.push(move)
            ply += 1

        game_lengths.append(ply)
        if training_log_path is not None:
            denom = max(1, game_prediction_samples)
            append_jsonl(
                training_log_path,
                build_event(
                    "training_game",
                    game_index=g + 1,
                    game_plies=ply,
                    observed_opponent_moves=game_observed_opponent_moves,
                    prediction_samples=game_prediction_samples,
                    top1_accuracy=game_top1_hits / denom,
                    top3_accuracy=game_top3_hits / denom,
                    avg_nll=game_nll_sum / denom,
                    avg_brier=game_brier_sum / denom,
                    result=board.result(claim_draw=True),
                ),
            )

    avg_len = float(sum(game_lengths) / max(1, len(game_lengths)))
    metric_denom = max(1, prediction_samples)
    return {
        "games": float(games),
        "observed_opponent_moves": float(observed_opponent_moves),
        "avg_game_plies": avg_len,
        "prediction_samples": float(prediction_samples),
        "top1_accuracy": float(top1_hits / metric_denom),
        "top3_accuracy": float(top3_hits / metric_denom),
        "avg_nll": float(nll_sum / metric_denom),
        "avg_brier": float(brier_sum / metric_denom),
    }


def top_expected_moves(
    board: chess.Board,
    model: AdaptiveMoveModel,
    profile: PlayerProfile,
    tell: VideoTellSnapshot,
    k: int = 3,
) -> List[Tuple[chess.Move, float]]:
    dist = model.predict_distribution(board, profile, tell)
    ranked = sorted(dist.items(), key=lambda x: x[1], reverse=True)
    return ranked[:k]


def ranked_moves_payload(
    board: chess.Board,
    ranked_moves: Sequence[Tuple[chess.Move, float]],
) -> List[Dict[str, object]]:
    payload: List[Dict[str, object]] = []
    for move, prob in ranked_moves:
        payload.append(
            {
                "move_san": board.san(move),
                "move_uci": move.uci(),
                "prob": float(prob),
            }
        )
    return payload


def shortest_mate_line_under_expected_opponent(
    board: chess.Board,
    model: AdaptiveMoveModel,
    profile: PlayerProfile,
    us_color: chess.Color,
    max_plies: int,
    tell: Optional[VideoTellSnapshot] = None,
    tell_provider: Optional[Callable[[int], VideoTellSnapshot]] = None,
    opponent_branch_k: int = 2,
) -> Optional[List[chess.Move]]:
    """Find shortest robust line to mate under modeled opponent uncertainty.

    On opponent nodes, we branch over top-k expected replies and only accept a line
    if all considered branches still have a mating continuation.
    On our nodes, we choose the move that yields the shortest successful mate line.
    """
    if tell_provider is None:
        if tell is None:
            raise ValueError("Either tell or tell_provider must be provided.")
        tell_provider = lambda _ply: tell

    branch_k = max(1, int(opponent_branch_k))
    memo: Dict[Tuple[str, int, int], Optional[List[chess.Move]]] = {}

    def rec(pos: chess.Board, depth: int) -> Optional[List[chess.Move]]:
        key = (pos.fen(), depth, pos.ply())
        if key in memo:
            return memo[key]

        if pos.is_checkmate():
            if pos.turn == us_color:
                memo[key] = None
                return None
            memo[key] = []
            return []

        if depth == 0 or pos.is_stalemate() or pos.is_insufficient_material():
            memo[key] = None
            return None

        if pos.turn == us_color:
            best: Optional[List[chess.Move]] = None
            for move in pos.legal_moves:
                pos.push(move)
                tail = rec(pos, depth - 1)
                pos.pop()
                if tail is None:
                    continue
                candidate = [move] + tail
                if best is None or len(candidate) < len(best):
                    best = candidate
            memo[key] = best
            return best

        tell_now = tell_provider(pos.ply())
        ranked = top_expected_moves(pos, model, profile, tell_now, k=branch_k)
        if not ranked:
            memo[key] = None
            return None

        worst_case: Optional[List[chess.Move]] = None
        for expected_move, _ in ranked:
            pos.push(expected_move)
            tail = rec(pos, depth - 1)
            pos.pop()
            if tail is None:
                memo[key] = None
                return None
            candidate = [expected_move] + tail
            if worst_case is None or len(candidate) > len(worst_case):
                worst_case = candidate

        memo[key] = worst_case
        return worst_case

    return rec(board.copy(stack=False), max_plies)


def pick_agent_move(
    board: chess.Board,
    model: AdaptiveMoveModel,
    profile: PlayerProfile,
    tell: VideoTellSnapshot,
    us_color: chess.Color,
    max_mate_plies: int,
    rng: random.Random,
    tell_provider: Optional[Callable[[int], VideoTellSnapshot]] = None,
    opponent_branch_k: int = 2,
) -> chess.Move:
    # Prefer immediate mate.
    for move in board.legal_moves:
        board.push(move)
        is_mate = board.is_checkmate()
        board.pop()
        if is_mate:
            return move

    line = shortest_mate_line_under_expected_opponent(
        board,
        model,
        profile,
        us_color=us_color,
        max_plies=max_mate_plies,
        tell=tell,
        tell_provider=tell_provider,
        opponent_branch_k=opponent_branch_k,
    )
    if line:
        return line[0]

    # Fallback: basic tactical preference.
    moves = list(board.legal_moves)
    checks = [m for m in moves if board.gives_check(m)]
    if checks and rng.random() < 0.65:
        return rng.choice(checks)

    captures = [m for m in moves if board.is_capture(m)]
    if captures and rng.random() < 0.65:
        return rng.choice(captures)

    return rng.choice(moves)


def san_line(board: chess.Board, line: Sequence[chess.Move]) -> List[str]:
    temp = board.copy(stack=False)
    out: List[str] = []
    for mv in line:
        out.append(temp.san(mv))
        temp.push(mv)
    return out


def run_inference_demo(
    model: AdaptiveMoveModel,
    profile: PlayerProfile,
    max_demo_plies: int,
    max_mate_plies: int,
    seed: int,
    inference_log_path: Optional[Path] = None,
    opponent_branch_k: int = 2,
) -> Dict[str, object]:
    rng = random.Random(seed + 999)
    true_policy = TrueOpponentPolicy()
    feed = MockVideoFeed(profile=profile, seed=seed + 555, interval_s=10)
    tell_provider = lambda ply: feed.snapshot_at(ply * 10.0)

    board = chess.Board()
    us_color = chess.WHITE

    print("\n=== Inference Demo (agent=White, modeled opponent=Black) ===")
    ply = 0
    while not board.is_game_over() and ply < max_demo_plies:
        tell = tell_provider(ply)
        fen_before = board.fen()

        if board.turn == us_color:
            line = shortest_mate_line_under_expected_opponent(
                board,
                model,
                profile,
                us_color=us_color,
                max_plies=max_mate_plies,
                tell=tell,
                tell_provider=tell_provider,
                opponent_branch_k=opponent_branch_k,
            )
            line_san = san_line(board, line) if line else []

            print(f"\n[State ply={ply}] turn=White")
            print(f"FEN: {board.fen()}")
            if line:
                print(f"Shortest robust mate line (top-{max(1, opponent_branch_k)} opp branches): {' '.join(line_san)}")
            else:
                print("Shortest mate line: none within search horizon")

            move = pick_agent_move(
                board,
                model,
                profile,
                tell,
                us_color=us_color,
                max_mate_plies=max_mate_plies,
                rng=rng,
                tell_provider=tell_provider,
                opponent_branch_k=opponent_branch_k,
            )
            agent_move_san = board.san(move)
            print(f"Agent move: {agent_move_san}")
            board.push(move)

            # Show expected next opponent moves after our action.
            expected_top3_after_move: List[Dict[str, object]] = []
            if not board.is_game_over():
                tell_next = tell_provider(ply + 1)
                top3 = top_expected_moves(board, model, profile, tell_next, k=3)
                if top3:
                    formatted = ", ".join(f"{board.san(mv)}:{prob:.2f}" for mv, prob in top3)
                    print(f"Expected opponent next move distribution (top3): {formatted}")
                    expected_top3_after_move = ranked_moves_payload(board, top3)

            if inference_log_path is not None:
                append_jsonl(
                    inference_log_path,
                    build_event(
                        "inference_ply",
                        ply=ply,
                        turn="white",
                        fen=fen_before,
                        opponent_branch_k=max(1, int(opponent_branch_k)),
                        shortest_mate_line_san=line_san,
                        chosen_move_san=agent_move_san,
                        chosen_move_uci=move.uci(),
                        expected_opponent_top3_after_move=expected_top3_after_move,
                    ),
                )

        else:
            top3 = top_expected_moves(board, model, profile, tell, k=3)
            expected_top3 = ranked_moves_payload(board, top3)
            print(f"\n[State ply={ply}] turn=Black")
            print(f"FEN: {board.fen()}")
            if top3:
                formatted = ", ".join(f"{board.san(mv)}:{prob:.2f}" for mv, prob in top3)
                print(f"Expected opponent move distribution (top3): {formatted}")

            # Actual opponent move comes from hidden policy (can differ from expectation).
            opp_move = true_policy.choose_move(board, profile, tell, rng)
            opp_move_san = board.san(opp_move)
            print(f"Actual opponent move: {opp_move_san}")

            # Online adaptation during inference.
            model.update(board, opp_move, tell)
            board.push(opp_move)

            if inference_log_path is not None:
                append_jsonl(
                    inference_log_path,
                    build_event(
                        "inference_ply",
                        ply=ply,
                        turn="black",
                        fen=fen_before,
                        expected_opponent_top3=expected_top3,
                        actual_opponent_move_san=opp_move_san,
                        actual_opponent_move_uci=opp_move.uci(),
                    ),
                )

        ply += 1

    result = board.result(claim_draw=True)
    outcome = board.outcome(claim_draw=True)
    print("\n=== Demo Finished ===")
    print(f"Final result: {result}")
    print(f"Termination: {outcome}")
    return {
        "final_result": result,
        "plies_played": float(ply),
        "termination": str(outcome),
        "is_game_over": bool(board.is_game_over()),
    }


def model_snapshot_payload(model: AdaptiveMoveModel) -> Dict[str, Dict[str, float]]:
    snapshot: Dict[str, Dict[str, float]] = {}
    for ctx in sorted(model.context_counts.keys()):
        counts = model.context_counts[ctx]
        snapshot[ctx] = {label: float(counts.get(label, 0.0)) for label in MOVE_LABELS}
    return snapshot


def model_move_snapshot_payload(model: AdaptiveMoveModel, top_n: int = 10) -> Dict[str, List[Dict[str, object]]]:
    snapshot: Dict[str, List[Dict[str, object]]] = {}
    limit = max(1, int(top_n))
    for ctx in sorted(model.context_move_counts.keys()):
        move_counts = model.context_move_counts[ctx]
        ranked = sorted(move_counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
        snapshot[ctx] = [{"move_uci": uci, "count": float(count)} for uci, count in ranked]
    return snapshot


def artifact_path_map(artifacts: RunArtifacts) -> Dict[str, str]:
    return {
        "run_meta.json": str(artifacts.run_meta),
        "training_games.jsonl": str(artifacts.training_games),
        "inference_trace.jsonl": str(artifacts.inference_trace),
        "run_summary.json": str(artifacts.run_summary),
        "model_snapshot.json": str(artifacts.model_snapshot),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple adaptive chess with mock tells")
    parser.add_argument("--profile", type=str, default="balanced", choices=sorted(DEFAULT_PROFILES.keys()))
    parser.add_argument("--train-games", type=int, default=300)
    parser.add_argument("--max-train-plies", type=int, default=80)
    parser.add_argument("--max-demo-plies", type=int, default=40)
    parser.add_argument("--max-mate-plies", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=str, default=DEFAULT_LOG_DIR)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument(
        "--log-json",
        dest="log_json",
        action="store_true",
        default=True,
        help="Write JSON/JSONL artifacts to the run directory (default: enabled)",
    )
    parser.add_argument(
        "--no-log-json",
        dest="log_json",
        action="store_false",
        help="Disable JSON/JSONL artifact output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = DEFAULT_PROFILES[args.profile]
    artifacts: Optional[RunArtifacts] = None

    if args.log_json:
        artifacts = init_run_artifacts(args.log_dir, args.run_id)
        write_json(
            artifacts.run_meta,
            build_event(
                "run_meta",
                run_id=artifacts.run_id,
                run_dir=str(artifacts.run_dir),
                script_path=str(Path(__file__).resolve()),
                cli_args=vars(args),
                profile=asdict(profile),
            ),
        )

    print(f"Using profile: {profile.name}")
    print(
        "Profile params: "
        f"aggr={profile.aggression:.2f}, tact={profile.tacticality:.2f}, "
        f"solid={profile.solidness:.2f}, speed={profile.speed:.2f}, comp={profile.composure:.2f}"
    )
    if artifacts is not None:
        print(f"Run ID: {artifacts.run_id}")

    model = AdaptiveMoveModel()
    stats = train_adaptive_model(
        model=model,
        profile=profile,
        games=max(1, args.train_games),
        max_plies=max(10, args.max_train_plies),
        seed=args.seed,
        training_log_path=artifacts.training_games if artifacts else None,
    )

    print("\n=== Training Summary ===")
    print(f"Games: {int(stats['games'])}")
    print(f"Observed opponent moves: {int(stats['observed_opponent_moves'])}")
    print(f"Average plies/game: {stats['avg_game_plies']:.1f}")
    print(f"Top-1 accuracy: {stats['top1_accuracy']:.3f}")
    print(f"Top-3 accuracy: {stats['top3_accuracy']:.3f}")
    print(f"Avg NLL: {stats['avg_nll']:.3f}")
    print(f"Avg Brier: {stats['avg_brier']:.3f}")

    inference_summary = run_inference_demo(
        model=model,
        profile=profile,
        max_demo_plies=max(10, args.max_demo_plies),
        max_mate_plies=max(2, args.max_mate_plies),
        seed=args.seed,
        inference_log_path=artifacts.inference_trace if artifacts else None,
        opponent_branch_k=2,
    )

    if artifacts is not None:
        snapshot = model_snapshot_payload(model)
        move_snapshot = model_move_snapshot_payload(model, top_n=10)
        write_json(
            artifacts.model_snapshot,
            build_event(
                "model_snapshot",
                run_id=artifacts.run_id,
                contexts=snapshot,
                context_top_moves=move_snapshot,
                context_count=len(snapshot),
                move_context_count=len(move_snapshot),
            ),
        )

        artifact_paths = artifact_path_map(artifacts)
        write_json(
            artifacts.run_summary,
            build_event(
                "run_summary",
                run_id=artifacts.run_id,
                profile=profile.name,
                training=stats,
                inference=inference_summary,
                artifact_paths=artifact_paths,
            ),
        )

        print("\nSaved artifacts:")
        for name, path in artifact_paths.items():
            print(f"- {name}: {path}")


if __name__ == "__main__":
    main()

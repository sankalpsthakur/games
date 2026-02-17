#!/usr/bin/env python3
"""Lightweight executable tests for adaptive_chess core behaviors."""

from __future__ import annotations

import inspect
import math
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping

import chess

import adaptive_chess as ac


class ScriptedBoard:
    """Minimal board-like object for deterministic game-tree testing."""

    def __init__(self, graph: Dict[str, Dict[str, Any]], node: str = "root") -> None:
        self._graph = graph
        self._node = node
        self._stack: List[tuple[str, int]] = []
        self._ply = int(self._graph[self._node].get("ply", 0))

    @property
    def turn(self) -> bool:
        return bool(self._graph[self._node].get("turn", chess.WHITE))

    @property
    def legal_moves(self) -> List[str]:
        return list(self._graph[self._node].get("edges", {}).keys())

    def push(self, move: str) -> None:
        edges = self._graph[self._node].get("edges", {})
        if move not in edges:
            raise AssertionError(f"Illegal scripted move {move!r} at node {self._node!r}")
        self._stack.append((self._node, self._ply))
        self._node = str(edges[move])
        self._ply += 1

    def pop(self) -> None:
        if not self._stack:
            raise AssertionError("ScriptedBoard pop() underflow")
        self._node, self._ply = self._stack.pop()

    def is_checkmate(self) -> bool:
        return bool(self._graph[self._node].get("checkmate", False))

    def is_stalemate(self) -> bool:
        return bool(self._graph[self._node].get("stalemate", False))

    def is_insufficient_material(self) -> bool:
        return bool(self._graph[self._node].get("insufficient_material", False))

    def fen(self) -> str:
        return self._node

    def ply(self) -> int:
        return self._ply

    def copy(self, stack: bool = False) -> "ScriptedBoard":
        out = ScriptedBoard(self._graph, node=self._node)
        out._ply = self._ply
        if stack:
            out._stack = list(self._stack)
        return out


def _tell(timestamp_s: float, confidence: float = 0.5) -> ac.VideoTellSnapshot:
    return ac.VideoTellSnapshot(
        timestamp_s=float(timestamp_s),
        gaze_aversion=0.3,
        blink_rate=0.3,
        jaw_tension=0.3,
        voice_stress=0.3,
        confidence=float(confidence),
    )


def _call_shortest(
    board: Any,
    *,
    max_plies: int,
    opponent_branch_k: int | None = None,
    tell_provider: Any | None = None,
):
    fn = ac.shortest_mate_line_under_expected_opponent
    sig = inspect.signature(fn)

    kwargs: Dict[str, Any] = {
        "board": board,
        "model": object(),
        "profile": ac.DEFAULT_PROFILES["balanced"],
        "tell": _tell(0.0),
        "us_color": chess.WHITE,
        "max_plies": int(max_plies),
    }

    if opponent_branch_k is not None:
        assert (
            "opponent_branch_k" in sig.parameters
        ), "shortest_mate_line_under_expected_opponent must accept opponent_branch_k"
        kwargs["opponent_branch_k"] = int(opponent_branch_k)

    if tell_provider is not None:
        assert "tell_provider" in sig.parameters, "shortest_mate_line_under_expected_opponent must accept tell_provider"
        kwargs["tell_provider"] = tell_provider

    return fn(**kwargs)


def test_shortest_mate_line_opponent_branch_k_guardrail() -> None:
    """If one likely opponent branch breaks mate, return None with branch_k > 1."""

    graph = {
        "root": {"turn": chess.WHITE, "edges": {"u": "opp"}},
        "opp": {"turn": chess.BLACK, "edges": {"allow": "mate_setup", "deny": "escape"}},
        "mate_setup": {"turn": chess.WHITE, "edges": {"mate": "mate"}},
        "mate": {"turn": chess.BLACK, "checkmate": True, "edges": {}},
        "escape": {"turn": chess.WHITE, "stalemate": True, "edges": {}},
    }
    board = ScriptedBoard(graph)

    original_top_expected_moves = ac.top_expected_moves

    def fake_top_expected_moves(pos: ScriptedBoard, model: Any, profile: Any, tell: Any, k: int = 3):
        if pos.fen() == "opp":
            ranked = [("allow", 0.60), ("deny", 0.40)]
            return ranked[:k]
        return []

    ac.top_expected_moves = fake_top_expected_moves
    try:
        line = _call_shortest(board, max_plies=3, opponent_branch_k=2)
    finally:
        ac.top_expected_moves = original_top_expected_moves

    assert line is None, "Expected None: one opponent branch breaks mating line when opponent_branch_k=2"


def test_tell_provider_depth_updates_change_opponent_ranking() -> None:
    """Depth tell snapshots should influence expected-opponent ranking inside search."""

    graph = {
        "root": {"turn": chess.WHITE, "edges": {"u0": "opp1"}},
        "opp1": {"turn": chess.BLACK, "edges": {"o1": "us1"}},
        "us1": {"turn": chess.WHITE, "edges": {"u1": "opp2"}},
        "opp2": {"turn": chess.BLACK, "edges": {"safe": "dead_end", "blunder": "us_mate"}},
        "dead_end": {"turn": chess.WHITE, "stalemate": True, "edges": {}},
        "us_mate": {"turn": chess.WHITE, "edges": {"mate": "mate"}},
        "mate": {"turn": chess.BLACK, "checkmate": True, "edges": {}},
    }
    board = ScriptedBoard(graph)

    class DeterministicTellProvider:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, *args: Any, **kwargs: Any) -> ac.VideoTellSnapshot:
            self.calls += 1

            # Prefer explicit board/node-based mapping when available.
            for value in list(args) + list(kwargs.values()):
                if hasattr(value, "fen"):
                    node = value.fen()
                    if node == "opp2":
                        return _tell(20.0, confidence=0.9)
                    return _tell(0.0, confidence=0.1)

            # Fallback: deterministic time progression by call count.
            if self.calls == 1:
                return _tell(0.0, confidence=0.1)
            return _tell(20.0, confidence=0.9)

    provider = DeterministicTellProvider()

    original_top_expected_moves = ac.top_expected_moves

    def fake_top_expected_moves(pos: ScriptedBoard, model: Any, profile: Any, tell: ac.VideoTellSnapshot, k: int = 3):
        if pos.fen() == "opp1":
            return [("o1", 1.0)]
        if pos.fen() == "opp2":
            if tell.confidence >= 0.5:
                ranked = [("blunder", 0.9), ("safe", 0.1)]
            else:
                ranked = [("safe", 0.9), ("blunder", 0.1)]
            return ranked[:k]
        return []

    ac.top_expected_moves = fake_top_expected_moves
    try:
        line = _call_shortest(board, max_plies=5, tell_provider=provider, opponent_branch_k=1)
    finally:
        ac.top_expected_moves = original_top_expected_moves

    assert provider.calls >= 2, "tell_provider should be consulted across depth"
    assert line is not None, "Expected non-empty mating line when deeper tell snapshot changes ranking to blunder"
    assert "blunder" in line, "Line should include branch selected under updated deeper tell snapshot"


def _extract_metric(stats: Dict[str, float], names: List[str]) -> float:
    for name in names:
        if name in stats:
            value = float(stats[name])
            assert math.isfinite(value), f"Metric {name} is not finite: {value}"
            return value
    raise AssertionError(f"Missing expected metric key. Looked for: {names}")


def test_training_metric_bookkeeping_sanity() -> None:
    model = ac.AdaptiveMoveModel()
    profile = ac.DEFAULT_PROFILES["balanced"]

    stats = ac.train_adaptive_model(
        model=model,
        profile=profile,
        games=2,
        max_plies=12,
        seed=7,
    )

    top1 = _extract_metric(stats, ["top1", "top1_accuracy", "top1_acc"])
    top3 = _extract_metric(stats, ["top3", "top3_accuracy", "top3_acc"])
    nll = _extract_metric(stats, ["nll", "avg_nll", "mean_nll"])
    brier = _extract_metric(stats, ["brier", "brier_score", "avg_brier"])

    assert 0.0 <= top1 <= 1.0, f"top1 out of range: {top1}"
    assert 0.0 <= top3 <= 1.0, f"top3 out of range: {top3}"
    assert nll >= 0.0, f"nll must be >= 0, got {nll}"
    assert brier >= 0.0, f"brier must be >= 0, got {brier}"


def test_move_specific_counts_influence_same_label_ranking() -> None:
    """Repeatedly observed move should outrank sibling move with same label."""

    model = ac.AdaptiveMoveModel()
    profile = ac.DEFAULT_PROFILES["balanced"]
    board = chess.Board()
    tell = _tell(0.0, confidence=0.7)

    favored = chess.Move.from_uci("g1f3")
    other = chess.Move.from_uci("b1c3")

    assert favored in board.legal_moves, "Expected g1f3 legal in initial position"
    assert other in board.legal_moves, "Expected b1c3 legal in initial position"
    assert model.move_label(board, favored) == model.move_label(board, other), (
        "Precondition failed: moves must share the same label"
    )

    dist_before = model.predict_distribution(board, profile, tell)
    favored_before = float(dist_before[favored])

    for _ in range(24):
        model.update(board, favored, tell)

    dist_after = model.predict_distribution(board, profile, tell)
    favored_after = float(dist_after[favored])
    other_after = float(dist_after[other])

    assert favored_after > other_after, (
        "Move-specific updates should raise favored move above same-label alternative"
    )
    assert favored_after > favored_before, "Favored move probability should increase after repeated updates"


def _move_count_summary_fields(payload: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    def rec(node: Any, path: str) -> None:
        if not isinstance(node, Mapping):
            return
        for key, value in node.items():
            key_str = str(key)
            key_l = key_str.lower()
            child_path = f"{path}.{key_str}" if path else key_str
            if "move" in key_l and ("count" in key_l or "total" in key_l or "summary" in key_l):
                out[child_path] = value
            rec(value, child_path)

    rec(payload, "")
    return out


def _flatten_numeric_values(node: Any) -> List[float]:
    out: List[float] = []
    if isinstance(node, (int, float)):
        out.append(float(node))
    elif isinstance(node, Mapping):
        for value in node.values():
            out.extend(_flatten_numeric_values(value))
    elif isinstance(node, (list, tuple)):
        for value in node:
            out.extend(_flatten_numeric_values(value))
    return out


def test_model_snapshot_includes_move_count_summary_fields() -> None:
    model = ac.AdaptiveMoveModel()
    board = chess.Board()
    tell = _tell(0.0, confidence=0.6)

    favored = chess.Move.from_uci("g1f3")
    other = chess.Move.from_uci("b1c3")
    for _ in range(6):
        model.update(board, favored, tell)
    model.update(board, other, tell)

    context_snapshot = ac.model_snapshot_payload(model)
    assert isinstance(context_snapshot, dict), "model_snapshot_payload must return a dictionary payload"

    move_snapshot_fn = getattr(ac, "model_move_snapshot_payload", None)
    assert callable(move_snapshot_fn), "Expected model_move_snapshot_payload helper for move-count summaries"
    context_top_moves = move_snapshot_fn(model, top_n=10)
    assert isinstance(context_top_moves, dict), "model_move_snapshot_payload must return a dictionary payload"

    snapshot_payload = ac.build_event(
        "model_snapshot",
        run_id="unit_test",
        contexts=context_snapshot,
        context_top_moves=context_top_moves,
        context_count=len(context_snapshot),
        move_context_count=len(context_top_moves),
    )

    move_summary_fields = _move_count_summary_fields(snapshot_payload)
    assert move_summary_fields, (
        "Expected snapshot to expose move-count summary field(s); "
        f"top-level keys were: {sorted(snapshot_payload.keys())}"
    )

    numeric_values: List[float] = []
    for value in move_summary_fields.values():
        numeric_values.extend(_flatten_numeric_values(value))

    assert numeric_values, "Move-count summary fields should contain numeric values"
    assert all(math.isfinite(v) and v >= 0.0 for v in numeric_values), (
        "Move-count summary values must be finite and non-negative"
    )


def _discover_tactical_benchmark_helper(chess_eval_module: Any) -> Any | None:
    preferred_names = (
        "run_tactical_benchmark",
        "tactical_benchmark",
        "tactical_benchmark_helper",
        "run_tactical_benchmark_helper",
    )
    for name in preferred_names:
        candidate = getattr(chess_eval_module, name, None)
        if callable(candidate):
            return candidate

    for name, candidate in inspect.getmembers(chess_eval_module, callable):
        name_l = name.lower()
        if "tactic" in name_l and ("benchmark" in name_l or "bench" in name_l):
            return candidate
    return None


def _looks_like_rate_key(key: str) -> bool:
    key_l = key.lower()
    if "accuracy" in key_l or key_l.endswith("_acc"):
        return True
    if "rate" in key_l and "count" not in key_l:
        return True
    return key_l in {"top1", "top3"}


def _collect_rate_values(payload: Any, path: str = "") -> Dict[str, float]:
    rates: Dict[str, float] = {}

    if isinstance(payload, Mapping):
        for key, value in payload.items():
            key_s = str(key)
            child = f"{path}.{key_s}" if path else key_s
            if isinstance(value, (int, float)) and _looks_like_rate_key(key_s):
                rates[child] = float(value)
            rates.update(_collect_rate_values(value, child))
        return rates

    if isinstance(payload, (list, tuple)):
        for idx, value in enumerate(payload):
            child = f"{path}[{idx}]"
            rates.update(_collect_rate_values(value, child))
        return rates

    if hasattr(payload, "__dict__"):
        rates.update(_collect_rate_values(vars(payload), path))
        for name, method in inspect.getmembers(payload, predicate=callable):
            if name.startswith("_") or not _looks_like_rate_key(name):
                continue
            try:
                sig = inspect.signature(method)
            except (TypeError, ValueError):
                continue
            if any(
                p.default is inspect.Parameter.empty
                and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                for p in sig.parameters.values()
            ):
                continue
            try:
                value = method()
            except Exception:
                continue
            if isinstance(value, (int, float)):
                child = f"{path}.{name}" if path else name
                rates[child] = float(value)
    return rates


def _invoke_tactical_benchmark_helper(helper: Any) -> Any:
    sig = inspect.signature(helper)
    kwargs: Dict[str, Any] = {}

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for name, param in sig.parameters.items():
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue

            name_l = name.lower()
            if "profile" in name_l:
                kwargs[name] = ac.DEFAULT_PROFILES.get("tactical", ac.DEFAULT_PROFILES["balanced"])
            elif name_l == "model":
                model = ac.AdaptiveMoveModel()
                ac.train_adaptive_model(
                    model=model,
                    profile=ac.DEFAULT_PROFILES.get("tactical", ac.DEFAULT_PROFILES["balanced"]),
                    games=1,
                    max_plies=12,
                    seed=5,
                )
                kwargs[name] = model
            elif name_l in {"seed", "seed_start"}:
                kwargs[name] = 5
            elif "seeds" in name_l:
                kwargs[name] = 1
            elif "train_games" in name_l:
                kwargs[name] = 1
            elif "max_train_plies" in name_l:
                kwargs[name] = 12
            elif name_l in {"max_plies"}:
                kwargs[name] = 4
            elif "max_demo_plies" in name_l or name_l in {"eval_plies"}:
                kwargs[name] = 8
            elif "max_mate_plies" in name_l:
                kwargs[name] = 4
            elif name_l in {"output", "output_path"} or ("output" in name_l and "path" in name_l):
                kwargs[name] = tmp_path / "tactical_benchmark.json"
            elif name_l in {"log_dir"}:
                kwargs[name] = str(tmp_path)
            elif name_l in {"run_id"}:
                kwargs[name] = "test_tactical_benchmark"
            elif name_l in {"log_json", "write_json", "verbose", "print_summary"}:
                kwargs[name] = False
            elif param.default is inspect.Parameter.empty:
                raise AssertionError(
                    f"Cannot invoke tactical benchmark helper; unsupported required parameter: {name}"
                )

        return helper(**kwargs)


def test_tactical_benchmark_helper_rates_are_finite_if_exposed() -> None:
    try:
        import chess_eval as ce
    except Exception as exc:
        print(f"[SKIP] chess_eval import failed: {exc}")
        return

    helper = _discover_tactical_benchmark_helper(ce)
    if helper is None:
        print("[SKIP] No tactical benchmark helper exposed by chess_eval")
        return

    result = _invoke_tactical_benchmark_helper(helper)
    rates = _collect_rate_values(result)
    if not rates and isinstance(result, dict):
        positions = result.get("positions")
        found = result.get("found")
        if isinstance(positions, (int, float)) and isinstance(found, (int, float)) and positions:
            rates["derived_found_rate"] = float(found) / float(positions)

    assert rates, "Tactical benchmark helper result must include at least one rate/accuracy metric"
    for key, value in rates.items():
        assert math.isfinite(value), f"Non-finite metric for {key}: {value}"
        assert 0.0 <= value <= 1.0, f"Metric {key} out of [0,1]: {value}"


def main() -> None:
    tests = [
        test_shortest_mate_line_opponent_branch_k_guardrail,
        test_tell_provider_depth_updates_change_opponent_ranking,
        test_training_metric_bookkeeping_sanity,
        test_move_specific_counts_influence_same_label_ranking,
        test_model_snapshot_includes_move_count_summary_fields,
        test_tactical_benchmark_helper_rates_are_finite_if_exposed,
    ]

    failures = 0
    for test in tests:
        print(f"[RUN] {test.__name__}")
        try:
            test()
        except Exception:
            failures += 1
            print(f"[FAIL] {test.__name__}")
            traceback.print_exc()
        else:
            print(f"[PASS] {test.__name__}")

    if failures:
        print(f"\n{failures}/{len(tests)} tests failed")
        raise SystemExit(1)

    print(f"\nAll {len(tests)} tests passed")


if __name__ == "__main__":
    main()

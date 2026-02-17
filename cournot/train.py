from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.runner import run_game

DEFAULT_ALGORITHMS = "PPO,A2C,DQN"


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value for --video.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate Cournot.")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--algorithms", type=str, default=DEFAULT_ALGORITHMS)
    parser.add_argument("--train-episodes", type=int, default=300)
    parser.add_argument("--eval-episodes", type=int, default=250)
    parser.add_argument("--video", type=parse_bool, default=True)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    algorithms = [name.strip() for name in args.algorithms.split(",") if name.strip()]
    run_game(
        game_name="cournot",
        seeds=args.seeds,
        algorithms=algorithms,
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        video=args.video,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

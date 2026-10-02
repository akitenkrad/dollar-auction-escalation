"""Independent exact solver and cross-language test-vector generator."""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path


def proposition_first_bid(s: int, b: int) -> int:
    if s < 2 or b < 1:
        raise ValueError("s must be at least 2 and b must be positive")
    return (b - 1) % (s - 1) + 1


def solve(s: int, b: int) -> tuple[int, int | None]:
    if s < 2 or b < 1:
        raise ValueError("s must be at least 2 and b must be positive")

    @lru_cache(maxsize=None)
    def value(me: int, opp: int) -> tuple[int, int, int | None]:
        best: tuple[int, int, int | None] = (
            -me,
            s - opp if opp > 0 else 0,
            None,
        )
        for amount in range(opp + 1, b + 1):
            opponent_payoff, mover_payoff, _ = value(opp, amount)
            if mover_payoff > best[0]:
                best = (mover_payoff, opponent_payoff, amount)
        return best

    first = value(0, 0)[2]
    if first is None:
        raise RuntimeError("the first mover unexpectedly dropped")
    return first, value(0, first)[2]


def generate_vectors(path: Path) -> None:
    rows: list[dict[str, int | None]] = []
    for s in (3, 5, 10):
        for b in range(1, 4 * s + 1):
            first, reply = solve(s, b)
            rows.append({"s": s, "b": b, "x1": first, "second_reply": reply})
    for b in (100, 250):
        first, reply = solve(100, b)
        rows.append({"s": 100, "b": b, "x1": first, "second_reply": reply})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    solve_parser = subparsers.add_parser("solve")
    solve_parser.add_argument("--s", type=int, required=True)
    solve_parser.add_argument("--b", type=int, required=True)
    vectors_parser = subparsers.add_parser("gen-vectors")
    vectors_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "solve":
        print(json.dumps({"x1": solve(args.s, args.b)[0], "second_reply": solve(args.s, args.b)[1]}))
    else:
        generate_vectors(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Command-line entry point for solver and analysis tools."""

from __future__ import annotations

import argparse
from pathlib import Path

from dollar_auction_tools.solver_ref import generate_vectors, solve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    solve_parser = commands.add_parser("solve")
    solve_parser.add_argument("--s", type=int, required=True)
    solve_parser.add_argument("--b", type=int, required=True)
    vectors = commands.add_parser("gen-vectors")
    vectors.add_argument("--output", type=Path, required=True)
    analyze = commands.add_parser("analyze")
    analyze.add_argument("--sweep", type=Path, required=True)
    analyze.add_argument("--quiz", type=Path)
    analyze.add_argument("--qre", type=Path)
    analyze.add_argument("--out", type=Path, required=True)
    reasons = commands.add_parser("code-reasons")
    reason_commands = reasons.add_subparsers(dest="reason_command", required=True)
    export = reason_commands.add_parser("export")
    export.add_argument("--run", type=Path, required=True)
    export.add_argument("--out", type=Path, required=True)
    agree = reason_commands.add_parser("agreement")
    agree.add_argument("coder_a", type=Path)
    agree.add_argument("coder_b", type=Path)
    visualize = commands.add_parser("visualize")
    visualize.add_argument("analysis_dir", type=Path)
    args = parser.parse_args(argv)

    if args.command == "solve":
        first, reply = solve(args.s, args.b)
        print(f'{{"x1": {first}, "second_reply": {reply if reply is not None else "null"}}}')
    elif args.command == "gen-vectors":
        generate_vectors(args.output)
    elif args.command == "analyze":
        from dollar_auction_tools.analyze import analyze_sweep

        analyze_sweep(args.sweep, out=args.out, quiz=args.quiz, qre=args.qre)
    elif args.command == "code-reasons" and args.reason_command == "export":
        from dollar_auction_tools.code_reasons import export_reasons

        export_reasons(args.run, args.out)
    elif args.command == "code-reasons":
        from dollar_auction_tools.code_reasons import agreement

        agreement(args.coder_a, args.coder_b)
    else:
        from dollar_auction_tools.visualize import redraw_plots

        redraw_plots(args.analysis_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

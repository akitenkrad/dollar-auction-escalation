"""dollar-auction-tools — 統合 CLI ディスパッチャ．

Usage:
    dollar-auction-tools visualize [...]
    dollar-auction-tools visualize-sweep [...]
    dollar-auction-tools show-experiment-settings [...]
"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="dollar-auction-tools",
        description="Visualization & analysis tools",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("visualize", help="単一実行結果の可視化", add_help=False)
    subparsers.add_parser(
        "visualize-sweep",
        help="スイープ結果の可視化 (パラメータ依存図 + 組み合わせ別グリッドアニメーション)",
        add_help=False,
    )
    subparsers.add_parser(
        "show-experiment-settings",
        help="実行結果ディレクトリの設定値を表示 (config.json / sweep_config.json)",
        add_help=False,
    )

    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in {"-h", "--help"}:
        parser.parse_args(argv)
        return

    command = argv[0]
    rest = argv[1:]
    if command == "visualize":
        from dollar_auction_tools.visualize import main as run_main
        run_main(rest)
    elif command == "visualize-sweep":
        from dollar_auction_tools.visualize_sweep import main as run_main
        run_main(rest)
    elif command == "show-experiment-settings":
        from dollar_auction_tools.show_experiment_settings import main as run_main
        run_main(rest)
    else:
        parser.parse_args(argv)  # 不正なサブコマンドはここでエラーになる


if __name__ == "__main__":
    main()

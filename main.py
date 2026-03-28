from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from integrated_alpha.common.config import ProjectPaths, SplitConfig
from integrated_alpha.common.io_utils import load_json, save_json, save_text
from integrated_alpha.data_module.panel_data import PanelDataManager
from integrated_alpha.data_module.tushare_sync import TushareSyncConfig, run_tushare_sync
from integrated_alpha.llm_module.claude_chat import ClaudeChatSession
from integrated_alpha.lstm_module.model import LSTMConfig, PriceDemoConfig, run_lstm_pipeline, run_price_demo_pipeline
from integrated_alpha.rl_module.symbolic_factor_agent import RLSearchConfig, run_rl_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Alpha Stock main entrypoint")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary_parser = subparsers.add_parser("data-summary", help="Show dataset summary")
    summary_parser.add_argument("--force-reload", action="store_true", help="Ignore cached panel file")

    tushare_parser = subparsers.add_parser("tushare-sync", help="Rebuild the local dataset from Tushare daily data")
    tushare_parser.add_argument("--limit-stocks", type=int, default=None)
    tushare_parser.add_argument("--start-date", type=int, default=20180101)
    tushare_parser.add_argument("--full-rebuild", action="store_true", help="Ignore local files and rebuild from scratch")
    tushare_parser.add_argument("--lookback-days", type=int, default=90, help="Days to refresh backwards in incremental mode")

    rl_parser = subparsers.add_parser("rl", help="Run only the RL module")
    add_shared_args(rl_parser)
    rl_parser.add_argument("--rl-episodes", type=int, default=20)

    lstm_parser = subparsers.add_parser("lstm", help="Run only the LSTM module")
    add_shared_args(lstm_parser)
    lstm_parser.add_argument("--lstm-stocks", type=int, default=20)
    lstm_parser.add_argument("--lstm-epochs", type=int, default=5)
    lstm_parser.add_argument("--sequence-length", type=int, default=20)

    price_demo_parser = subparsers.add_parser("lstm-price-demo", help="Run a single-stock price prediction demo")
    add_shared_args(price_demo_parser)
    price_demo_parser.add_argument("--stock-code", type=str, default="000001.SZ")
    price_demo_parser.add_argument("--lstm-epochs", type=int, default=20)
    price_demo_parser.add_argument("--sequence-length", type=int, default=30)

    run_all_parser = subparsers.add_parser("run-all", help="Run data summary, RL, and LSTM together")
    add_shared_args(run_all_parser)
    run_all_parser.add_argument("--rl-episodes", type=int, default=20)
    run_all_parser.add_argument("--lstm-stocks", type=int, default=20)
    run_all_parser.add_argument("--lstm-epochs", type=int, default=5)
    run_all_parser.add_argument("--sequence-length", type=int, default=20)
    run_all_parser.add_argument("--chat", action="store_true", help="Start the chat assistant after the pipeline")

    chat_parser = subparsers.add_parser("chat", help="Open the chat assistant")
    chat_parser.add_argument("--question", type=str, default="", help="Ask one question and exit")

    return parser


def add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-end", type=int, default=20221230)
    parser.add_argument("--val-end", type=int, default=20241231)
    parser.add_argument("--test-end", type=int, default=20251231)
    parser.add_argument("--force-reload", action="store_true", help="Ignore cached panel file")


def render_markdown_summary(summary: dict[str, Any]) -> str:
    data = summary["data_summary"]
    rl = summary["rl_summary"]
    lstm = summary["lstm_summary"]

    top_stock = rl["top_stocks"][0]["ts_code"] if rl["top_stocks"] else "N/A"
    return "\n".join(
        [
            "# Integrated Alpha Summary",
            "",
            "## Data",
            f"- Stocks: {data['stock_count']}",
            f"- Rows: {data['row_count']}",
            f"- Date range: {data['date_start']} to {data['date_end']}",
            f"- Latest trade date: {data['latest_trade_date']}",
            "",
            "## RL",
            f"- Best formula: `{rl['best_formula']}`",
            f"- Train Rank IC: {rl['train_rank_ic']:.6f}",
            f"- Validation Rank IC: {rl['val_rank_ic']:.6f}",
            f"- Test Rank IC: {rl['test_rank_ic']:.6f}",
            f"- Top stock on latest date: {top_stock}",
            "",
            "## LSTM",
            f"- Training stocks: {lstm['num_stocks']}",
            f"- Test RMSE: {lstm['rmse']:.6f}",
            f"- Test MAE: {lstm['mae']:.6f}",
            f"- Test daily Rank IC: {lstm['mean_daily_rank_ic']:.6f}",
            f"- Baseline daily Rank IC: {lstm['baseline_mean_daily_rank_ic']:.6f}",
        ]
    )


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    paths = ProjectPaths.from_root(PROJECT_ROOT)
    split_config = SplitConfig(train_end=args.train_end, val_end=args.val_end, test_end=args.test_end)

    manager = PanelDataManager(paths.data_dir, paths.cache_dir)
    panel = manager.load_panel(force_reload=args.force_reload)
    data_summary = manager.summarize(panel)

    rl_result = run_rl_pipeline(
        panel=panel,
        output_dir=paths.rl_dir,
        split_config=split_config,
        config=RLSearchConfig(episodes=args.rl_episodes, random_seed=args.seed),
    )

    lstm_stock_codes = manager.select_evenly_spaced_stock_codes(limit=args.lstm_stocks)
    lstm_panel = manager.filter_stocks(panel, lstm_stock_codes)
    lstm_result = run_lstm_pipeline(
        panel=lstm_panel,
        output_dir=paths.lstm_dir,
        split_config=split_config,
        config=LSTMConfig(
            random_seed=args.seed,
            epochs=args.lstm_epochs,
            sequence_length=args.sequence_length,
            stock_limit=args.lstm_stocks,
        ),
    )

    combined_summary = {
        "data_summary": data_summary,
        "rl_summary": rl_result["summary"],
        "lstm_summary": lstm_result["summary"],
        "artifact_paths": {
            "rl_summary": str(paths.rl_dir / "summary.json"),
            "lstm_summary": str(paths.lstm_dir / "summary.json"),
            "markdown_summary": str(paths.combined_dir / "project_summary.md"),
        },
    }
    save_json(combined_summary, paths.combined_dir / "experiment_summary.json")
    save_text(render_markdown_summary(combined_summary), paths.combined_dir / "project_summary.md")

    print("\nPipeline complete.")
    print(json.dumps(combined_summary, indent=2))
    return combined_summary


def run_rl_only(args: argparse.Namespace) -> dict[str, Any]:
    paths = ProjectPaths.from_root(PROJECT_ROOT)
    split_config = SplitConfig(train_end=args.train_end, val_end=args.val_end, test_end=args.test_end)
    manager = PanelDataManager(paths.data_dir, paths.cache_dir)
    panel = manager.load_panel(force_reload=args.force_reload)

    result = run_rl_pipeline(
        panel=panel,
        output_dir=paths.rl_dir,
        split_config=split_config,
        config=RLSearchConfig(episodes=args.rl_episodes, random_seed=args.seed),
    )
    print(json.dumps(result["summary"], indent=2))
    return result["summary"]


def run_lstm_only(args: argparse.Namespace) -> dict[str, Any]:
    paths = ProjectPaths.from_root(PROJECT_ROOT)
    split_config = SplitConfig(train_end=args.train_end, val_end=args.val_end, test_end=args.test_end)
    manager = PanelDataManager(paths.data_dir, paths.cache_dir)
    panel = manager.load_panel(force_reload=args.force_reload)
    lstm_stock_codes = manager.select_evenly_spaced_stock_codes(limit=args.lstm_stocks)
    lstm_panel = manager.filter_stocks(panel, lstm_stock_codes)

    result = run_lstm_pipeline(
        panel=lstm_panel,
        output_dir=paths.lstm_dir,
        split_config=split_config,
        config=LSTMConfig(
            random_seed=args.seed,
            epochs=args.lstm_epochs,
            sequence_length=args.sequence_length,
            stock_limit=args.lstm_stocks,
        ),
    )
    print(json.dumps(result["summary"], indent=2))
    return result["summary"]


def run_lstm_price_demo(args: argparse.Namespace) -> dict[str, Any]:
    paths = ProjectPaths.from_root(PROJECT_ROOT)
    split_config = SplitConfig(train_end=args.train_end, val_end=args.val_end, test_end=args.test_end)
    manager = PanelDataManager(paths.data_dir, paths.cache_dir)
    panel = manager.load_panel(stock_codes=[args.stock_code], force_reload=args.force_reload)

    result = run_price_demo_pipeline(
        panel=panel,
        output_dir=paths.lstm_dir,
        split_config=split_config,
        config=PriceDemoConfig(
            stock_code=args.stock_code,
            random_seed=args.seed,
            epochs=args.lstm_epochs,
            sequence_length=args.sequence_length,
        ),
    )
    print(json.dumps(result["summary"], indent=2))
    return result["summary"]


def run_data_summary(force_reload: bool) -> dict[str, Any]:
    paths = ProjectPaths.from_root(PROJECT_ROOT)
    manager = PanelDataManager(paths.data_dir, paths.cache_dir)
    panel = manager.load_panel(force_reload=force_reload)
    summary = manager.summarize(panel)
    save_json(summary, paths.combined_dir / "data_summary.json")
    print(json.dumps(summary, indent=2))
    return summary


def run_tushare_sync_command(args: argparse.Namespace) -> dict[str, Any]:
    try:
        summary = run_tushare_sync(
            project_root=PROJECT_ROOT,
            config=TushareSyncConfig(
                start_date=args.start_date,
                limit_stocks=args.limit_stocks,
                incremental=not args.full_rebuild,
                lookback_days=args.lookback_days,
            ),
        )
    except RuntimeError as exc:
        print(str(exc))
        return {}
    print(json.dumps(summary, indent=2))
    return summary


def start_chat(question: str = "") -> None:
    paths = ProjectPaths.from_root(PROJECT_ROOT)
    summary_path = paths.combined_dir / "experiment_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            "No combined experiment summary found. Run `main.py run-all` before using chat."
        )

    summary = load_json(summary_path)
    chat_session = ClaudeChatSession.from_env(output_dir=paths.llm_dir)

    if question:
        try:
            answer = chat_session.ask(question=question, experiment_summary=summary)
        except RuntimeError as exc:
            print(str(exc))
            return
        print(answer)
        return

    try:
        chat_session.interactive_loop(experiment_summary=summary)
    except RuntimeError as exc:
        print(str(exc))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "data-summary":
        run_data_summary(force_reload=args.force_reload)
    elif args.command == "tushare-sync":
        run_tushare_sync_command(args)
    elif args.command == "rl":
        run_rl_only(args)
    elif args.command == "lstm":
        run_lstm_only(args)
    elif args.command == "lstm-price-demo":
        run_lstm_price_demo(args)
    elif args.command == "run-all":
        run_pipeline(args)
        if args.chat:
            start_chat()
    elif args.command == "chat":
        start_chat(question=args.question)


if __name__ == "__main__":
    main()

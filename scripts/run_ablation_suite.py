from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from integrated_alpha.common.config import ProjectPaths, SplitConfig
from integrated_alpha.common.io_utils import ensure_directory, save_json, save_text
from integrated_alpha.common.metrics import mean_daily_rank_ic
from integrated_alpha.data_module.panel_data import PanelDataManager
from integrated_alpha.lstm_module.model import LSTMConfig, run_lstm_pipeline
from integrated_alpha.rl_module.symbolic_factor_agent import FormulaScorer, RLSearchConfig, run_rl_pipeline


def _daily_rank_pct(values: pd.Series) -> pd.Series:
    if len(values) <= 1:
        return pd.Series(np.ones(len(values)), index=values.index, dtype=float)
    return values.rank(method="average", pct=True)


def _combined_rank_score(frame: pd.DataFrame) -> pd.Series:
    grouped = frame.groupby("trade_date", group_keys=False)
    rl_rank = grouped["rl_score"].transform(_daily_rank_pct)
    lstm_rank = grouped["lstm_score"].transform(_daily_rank_pct)
    return 0.45 * rl_rank + 0.55 * lstm_rank


def main() -> None:
    paths = ProjectPaths.from_root(ROOT)
    output_dir = paths.output_dir / "ablation"
    ensure_directory(output_dir)

    split_config = SplitConfig()
    manager = PanelDataManager(paths.data_dir, paths.cache_dir)
    panel = manager.load_panel(force_reload=False)

    rl_config = RLSearchConfig(episodes=20, random_seed=42, top_n=40, pool_size=3)
    rl_result = run_rl_pipeline(
        panel=panel,
        output_dir=output_dir / "rl_main",
        split_config=split_config,
        config=rl_config,
    )

    candidate_codes = [row["ts_code"] for row in rl_result["summary"]["top_stocks"]]
    candidate_panel = manager.filter_stocks(panel, candidate_codes)

    lstm_result = run_lstm_pipeline(
        panel=candidate_panel,
        output_dir=output_dir / "lstm_candidates",
        split_config=split_config,
        config=LSTMConfig(
            epochs=5,
            sequence_length=20,
            stock_limit=len(candidate_codes),
            random_seed=42,
        ),
    )

    candidate_scorer = FormulaScorer(
        panel=candidate_panel,
        split_config=split_config,
        top_n=rl_config.top_n,
        config=rl_config,
    )
    rl_pool = rl_result["summary"]["factor_pool"]
    pool_formulas = [item["formula"] for item in rl_pool]
    pool_weights = np.array([float(item["weight"]) for item in rl_pool], dtype=np.float64)

    rl_test_frame = candidate_scorer.test[["ts_code", "trade_date", "future_return_20d"]].copy()
    rl_test_frame["rl_score"] = candidate_scorer._pool_factor(candidate_scorer.test, pool_formulas, pool_weights)

    lstm_test = lstm_result["prediction_frame"][["ts_code", "trade_date", "prediction", "target"]].copy()
    merged = rl_test_frame.merge(lstm_test, on=["ts_code", "trade_date"], how="inner")
    merged = merged.rename(columns={"prediction": "lstm_score", "future_return_20d": "rl_target"})
    merged["target"] = merged["target"].astype(float)
    merged["combined_score"] = _combined_rank_score(merged)

    rl_only_rank_ic = mean_daily_rank_ic(merged, prediction_col="rl_score", target_col="target")
    lstm_only_rank_ic = mean_daily_rank_ic(merged, prediction_col="lstm_score", target_col="target")
    combined_rank_ic = mean_daily_rank_ic(merged, prediction_col="combined_score", target_col="target")

    single_alpha = dict(rl_result["summary"]["representative_formula_stats"])
    pooled_alpha = {
        "train_rank_ic": float(rl_result["summary"]["train_rank_ic"]),
        "val_rank_ic": float(rl_result["summary"]["val_rank_ic"]),
        "test_rank_ic": float(rl_result["summary"]["test_rank_ic"]),
        "selection_score": float(rl_result["summary"]["selection_score"]),
        "pool_size": int(len(rl_pool)),
    }

    pool_rows: list[dict[str, float]] = []
    for pool_size in range(1, 6):
        sweep_config = RLSearchConfig(
            episodes=20,
            random_seed=42,
            top_n=40,
            pool_size=pool_size,
        )
        sweep_result = run_rl_pipeline(
            panel=panel,
            output_dir=output_dir / f"pool_size_{pool_size}",
            split_config=split_config,
            config=sweep_config,
        )
        pool_rows.append(
            {
                "pool_size": pool_size,
                "train_rank_ic": float(sweep_result["summary"]["train_rank_ic"]),
                "val_rank_ic": float(sweep_result["summary"]["val_rank_ic"]),
                "test_rank_ic": float(sweep_result["summary"]["test_rank_ic"]),
                "selection_score": float(sweep_result["summary"]["selection_score"]),
                "actual_pool_size": int(len(sweep_result["summary"]["factor_pool"])),
            }
        )

    pool_sensitivity = pd.DataFrame(pool_rows)
    pool_sensitivity.to_csv(output_dir / "pool_size_sensitivity.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(8, 4))
    plt.plot(pool_sensitivity["pool_size"], pool_sensitivity["val_rank_ic"], marker="o", label="Validation Rank IC")
    plt.plot(pool_sensitivity["pool_size"], pool_sensitivity["test_rank_ic"], marker="o", label="Test Rank IC")
    plt.xlabel("Pool Size")
    plt.ylabel("Rank IC")
    plt.title("RL Pool Size Sensitivity")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "pool_size_sensitivity.png", dpi=150)
    plt.close()

    summary = {
        "rl_vs_lstm_vs_combined": {
            "rl_only_rank_ic": float(rl_only_rank_ic),
            "lstm_only_rank_ic": float(lstm_only_rank_ic),
            "combined_rank_ic": float(combined_rank_ic),
            "evaluation_rows": int(len(merged)),
            "evaluation_stocks": int(merged["ts_code"].nunique()),
        },
        "single_alpha_vs_pooled_alpha": {
            "single_alpha": single_alpha,
            "pooled_alpha": pooled_alpha,
        },
        "pool_size_sensitivity": pool_rows,
    }
    save_json(summary, output_dir / "ablation_summary.json")

    markdown = "\n".join(
        [
            "# Ablation Summary",
            "",
            "## RL Only vs LSTM Only vs RL + LSTM",
            f"- RL only Rank IC: {rl_only_rank_ic:.6f}",
            f"- LSTM only Rank IC: {lstm_only_rank_ic:.6f}",
            f"- RL + LSTM Rank IC: {combined_rank_ic:.6f}",
            "",
            "## Single Alpha vs Pooled Alpha",
            f"- Single alpha formula: {single_alpha.get('formula', 'N/A')}",
            f"- Single alpha test Rank IC: {single_alpha.get('test_rank_ic', float('nan')):.6f}",
            f"- Pooled alpha test Rank IC: {pooled_alpha['test_rank_ic']:.6f}",
            "",
            "## Pool Size Sensitivity",
            *(f"- pool_size={row['pool_size']}: val={row['val_rank_ic']:.6f}, test={row['test_rank_ic']:.6f}" for row in pool_rows),
        ]
    )
    save_text(markdown, output_dir / "ablation_summary.md")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

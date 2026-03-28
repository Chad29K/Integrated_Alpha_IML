from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from integrated_alpha.common.config import ProjectPaths, SplitConfig
from integrated_alpha.common.io_utils import ensure_directory, load_json, save_json
from integrated_alpha.data_module.panel_data import FEATURE_COLUMNS, PanelDataManager
from integrated_alpha.lstm_module.model import (
    LSTMConfig,
    PriceDemoConfig,
    predict_latest_returns,
    run_lstm_pipeline,
    run_price_demo_pipeline,
)
from integrated_alpha.rl_module.symbolic_factor_agent import RLSearchConfig, run_rl_pipeline

FEATURE_LABELS = {
    "open_adj": "Open",
    "high_adj": "High",
    "low_adj": "Low",
    "close_adj": "Close",
    "vol": "Volume",
    "return_1d": "1-day return",
    "return_5d": "5-day return",
    "return_10d": "10-day return",
    "ma_5": "MA(5)",
    "ma_10": "MA(10)",
    "volatility_5": "5-day volatility",
    "intraday_strength": "Intraday strength",
    "ma_gap_5": "Price vs MA(5)",
    "ma_gap_10": "Price vs MA(10)",
    "trend_gap": "MA(5) vs MA(10)",
    "volume_ratio_5": "Volume ratio(5)",
    "volume_ratio_10": "Volume ratio(10)",
}


@dataclass(frozen=True)
class DashboardBuildConfig:
    candidate_count: int = 40
    top_n: int = 10
    rl_episodes: int = 20
    lstm_epochs: int = 6
    sequence_length: int = 20
    random_seed: int = 42


def build_dashboard_outputs(
    project_root: Path,
    force_refresh: bool = False,
    config: DashboardBuildConfig | None = None,
) -> dict[str, Any]:
    config = config or DashboardBuildConfig()
    paths = ProjectPaths.from_root(project_root)
    ensure_directory(paths.dashboard_dir)

    artifact_paths = _artifact_paths(paths.dashboard_dir)
    if not force_refresh and _artifacts_ready(artifact_paths):
        return _load_dashboard_outputs(project_root)

    manager = PanelDataManager(paths.data_dir, paths.cache_dir)
    panel = manager.load_panel(force_reload=force_refresh)
    split_config = _dashboard_split_config(panel)
    panel_with_rl_features = _augment_rl_features(panel)
    latest_market_snapshot = manager.latest_snapshot(panel_with_rl_features)

    rl_result = run_rl_pipeline(
        panel=panel,
        output_dir=paths.dashboard_dir / "rl_candidates",
        split_config=split_config,
        config=RLSearchConfig(
            episodes=config.rl_episodes,
            top_n=config.candidate_count,
            random_seed=config.random_seed,
        ),
    )
    candidate_frame = pd.DataFrame(rl_result["summary"]["top_stocks"]).copy()
    if candidate_frame.empty:
        raise RuntimeError("Dashboard build failed because RL did not return any candidate stocks.")

    candidate_frame = candidate_frame.rename(columns={"rank": "rl_rank"})
    candidate_codes = candidate_frame["ts_code"].tolist()
    candidate_panel = manager.filter_stocks(panel, candidate_codes)
    candidate_panel_with_rl_features = _augment_rl_features(candidate_panel)

    lstm_result = run_lstm_pipeline(
        panel=candidate_panel,
        output_dir=paths.dashboard_dir / "lstm_candidates",
        split_config=split_config,
        config=LSTMConfig(
            epochs=config.lstm_epochs,
            sequence_length=config.sequence_length,
            stock_limit=len(candidate_codes),
            random_seed=config.random_seed,
        ),
    )
    latest_predictions = predict_latest_returns(
        panel=candidate_panel,
        model=lstm_result["model"],
        scaler=lstm_result["scaler"],
        sequence_length=config.sequence_length,
    )
    if latest_predictions.empty:
        raise RuntimeError("Dashboard build failed because LSTM did not produce latest predictions.")

    candidate_latest_snapshot = (
        candidate_panel_with_rl_features.sort_values(["ts_code", "trade_date"])
        .groupby("ts_code", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    candidate_latest_snapshot = candidate_latest_snapshot.merge(
        candidate_frame[["ts_code", "trade_date", "factor_value", "rl_rank"]],
        on=["ts_code", "trade_date"],
        how="left",
    )

    recommendation_frame = candidate_frame.merge(
        latest_predictions,
        on=["ts_code", "trade_date"],
        how="inner",
    )
    recommendation_frame["rl_rank_score"] = _descending_rank_score(recommendation_frame["rl_rank"])
    recommendation_frame["lstm_rank_score"] = recommendation_frame["predicted_future_return_20d"].rank(
        method="average",
        pct=True,
    )
    recommendation_frame["combined_score"] = (
        0.55 * recommendation_frame["lstm_rank_score"] + 0.45 * recommendation_frame["rl_rank_score"]
    )
    recommendation_frame = recommendation_frame.sort_values(
        ["combined_score", "predicted_future_return_20d", "factor_value"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    recommendation_frame["recommendation_rank"] = recommendation_frame.index + 1
    top_recommendations = recommendation_frame.head(config.top_n).copy()

    summary = {
        "latest_trade_date": int(recommendation_frame["trade_date"].max()),
        "market_stock_count": int(latest_market_snapshot["ts_code"].nunique()),
        "candidate_count": int(len(recommendation_frame)),
        "top_n": int(config.top_n),
        "rl_formula": str(rl_result["summary"]["best_formula"]),
        "rl_factor_pool": rl_result["summary"].get("factor_pool", []),
        "rl_validation_rank_ic": float(rl_result["summary"]["val_rank_ic"]),
        "rl_test_rank_ic": float(rl_result["summary"]["test_rank_ic"]),
        "lstm_rmse": float(lstm_result["summary"]["rmse"]),
        "lstm_mean_daily_rank_ic": float(lstm_result["summary"]["mean_daily_rank_ic"]),
        "notes": [
            "Top 10 recommendations are ranked by a weighted combination of RL factor rank and latest LSTM return forecast.",
            "LSTM is trained on the RL candidate universe for dashboard speed and stability.",
        ],
    }

    top_recommendations.to_csv(artifact_paths["top_recommendations"], index=False, encoding="utf-8-sig")
    recommendation_frame.to_csv(artifact_paths["all_recommendations"], index=False, encoding="utf-8-sig")
    candidate_latest_snapshot.to_csv(artifact_paths["latest_snapshot"], index=False, encoding="utf-8-sig")
    latest_market_snapshot.to_csv(artifact_paths["market_snapshot"], index=False, encoding="utf-8-sig")
    lstm_result["prediction_frame"].to_csv(artifact_paths["test_predictions"], index=False, encoding="utf-8-sig")
    save_json(summary, artifact_paths["summary"])

    return _load_dashboard_outputs(project_root)


def get_dashboard_bundle(project_root: Path) -> dict[str, Any]:
    return _get_dashboard_bundle_cached(str(project_root))


def refresh_dashboard_bundle(project_root: Path) -> dict[str, Any]:
    _get_dashboard_bundle_cached.cache_clear()
    _get_price_demo_bundle_cached.cache_clear()
    return build_dashboard_outputs(project_root=project_root, force_refresh=True)


@lru_cache(maxsize=1)
def _get_dashboard_bundle_cached(project_root: str) -> dict[str, Any]:
    return build_dashboard_outputs(project_root=Path(project_root), force_refresh=False)


def get_price_demo_bundle(project_root: Path, stock_code: str) -> dict[str, Any]:
    return _get_price_demo_bundle_cached(str(project_root), stock_code)


@lru_cache(maxsize=16)
def _get_price_demo_bundle_cached(project_root: str, stock_code: str) -> dict[str, Any]:
    root = Path(project_root)
    paths = ProjectPaths.from_root(root)
    manager = PanelDataManager(paths.data_dir, paths.cache_dir)
    panel = manager.load_panel(stock_codes=[stock_code], force_reload=False)
    price_demo_split = SplitConfig(test_end=int(panel["trade_date"].max()))
    output_dir = paths.dashboard_dir / "price_demos" / stock_code.replace(".", "_")
    result = run_price_demo_pipeline(
        panel=panel,
        output_dir=output_dir,
        split_config=price_demo_split,
        config=PriceDemoConfig(
            stock_code=stock_code,
            epochs=16,
            patience=4,
            sequence_length=30,
            random_seed=42,
        ),
    )
    return {
        "summary": result["summary"],
        "prediction_frame": result["prediction_frame"].copy(),
        "output_dir": output_dir,
    }


def explain_stock_pick(bundle: dict[str, Any], stock_code: str) -> dict[str, Any]:
    recommendations = bundle["all_recommendations"].copy()
    latest_snapshot = bundle["latest_snapshot"].copy()
    market_snapshot = bundle["market_snapshot"].copy()
    test_predictions = bundle["test_predictions"].copy()
    panel = bundle["panel"].copy()

    rec_row = recommendations.loc[recommendations["ts_code"] == stock_code]
    if rec_row.empty:
        raise KeyError(f"Stock {stock_code} is not available in the current dashboard recommendation set.")
    rec = rec_row.iloc[0]

    latest_row = latest_snapshot.loc[latest_snapshot["ts_code"] == stock_code]
    latest = latest_row.iloc[0] if not latest_row.empty else panel.loc[panel["ts_code"] == stock_code].sort_values("trade_date").iloc[-1]
    formula = str(bundle["summary"]["rl_formula"])
    factor_pool = bundle["summary"].get("rl_factor_pool", [])
    left_feature = ""
    operator = ""
    right_feature = ""
    left_value = float("nan")
    right_value = float("nan")
    left_percentile = float("nan")
    right_percentile = float("nan")
    if len(factor_pool) <= 1:
        left_feature, operator, right_feature = _parse_formula(formula)
        left_value = float(latest.get(left_feature, np.nan))
        right_value = float(latest.get(right_feature, np.nan))
        left_percentile = _percentile_rank(market_snapshot[left_feature], left_value)
        right_percentile = _percentile_rank(market_snapshot[right_feature], right_value)
    latest_close = float(latest.get("close_adj", np.nan))
    predicted_return = float(rec["predicted_future_return_20d"])
    factor_value = float(rec["factor_value"])
    rl_rank = int(rec["rl_rank"])
    recommendation_rank = int(rec["recommendation_rank"])

    stock_fit_frame = (
        test_predictions.loc[test_predictions["ts_code"] == stock_code]
        .sort_values("trade_date")
        .reset_index(drop=True)
    )
    stock_price_history = (
        panel.loc[panel["ts_code"] == stock_code, ["trade_date", "close_adj"]]
        .sort_values("trade_date")
        .tail(260)
        .reset_index(drop=True)
    )

    reasons = [
        {
            "title": "RL factor signal",
            "body": _rl_reason_text(
                formula=formula,
                factor_value=factor_value,
                rl_rank=rl_rank,
                factor_pool=factor_pool,
                left_feature=left_feature,
                left_value=left_value,
                left_percentile=left_percentile,
                right_feature=right_feature,
                right_value=right_value,
                right_percentile=right_percentile,
                operator=operator,
            ),
        },
        {
            "title": "LSTM forecast",
            "body": (
                f"The LSTM model predicts a future 20-day return of {predicted_return:.2%} for {stock_code} "
                f"based on the most recent {bundle['config']['sequence_length']} trading-day sequence."
            ),
        },
        {
            "title": "Current market snapshot",
            "body": (
                f"The latest close price is {latest_close:.2f}. The stock is ranked #{recommendation_rank} in the "
                f"dashboard recommendation list after combining RL rank and LSTM forecast."
            ),
        },
    ]

    fit_metrics = _fit_metrics(stock_fit_frame)
    signal_snapshot = {
        FEATURE_LABELS.get(name, name): float(latest[name])
        for name in FEATURE_COLUMNS
        if name in latest.index and pd.notna(latest[name])
    }

    return {
        "stock_code": stock_code,
        "recommendation_rank": recommendation_rank,
        "predicted_future_return_20d": predicted_return,
        "factor_value": factor_value,
        "formula": formula,
        "reasons": reasons,
        "fit_metrics": fit_metrics,
        "fit_frame": stock_fit_frame,
        "price_history": stock_price_history,
        "signal_snapshot": signal_snapshot,
    }


def _artifacts_ready(paths: dict[str, Path]) -> bool:
    return all(path.exists() for path in paths.values())


def _artifact_paths(dashboard_dir: Path) -> dict[str, Path]:
    return {
        "summary": dashboard_dir / "summary.json",
        "top_recommendations": dashboard_dir / "top_recommendations.csv",
        "all_recommendations": dashboard_dir / "all_recommendations.csv",
        "latest_snapshot": dashboard_dir / "latest_snapshot.csv",
        "market_snapshot": dashboard_dir / "market_snapshot.csv",
        "test_predictions": dashboard_dir / "test_predictions.csv",
    }


def _load_dashboard_outputs(project_root: Path) -> dict[str, Any]:
    paths = ProjectPaths.from_root(project_root)
    artifact_paths = _artifact_paths(paths.dashboard_dir)
    manager = PanelDataManager(paths.data_dir, paths.cache_dir)
    panel = manager.load_panel(force_reload=False)
    return {
        "summary": load_json(artifact_paths["summary"]),
        "top_recommendations": pd.read_csv(artifact_paths["top_recommendations"]),
        "all_recommendations": pd.read_csv(artifact_paths["all_recommendations"]),
        "latest_snapshot": pd.read_csv(artifact_paths["latest_snapshot"]),
        "market_snapshot": pd.read_csv(artifact_paths["market_snapshot"]),
        "test_predictions": pd.read_csv(artifact_paths["test_predictions"]),
        "panel": panel,
        "config": {
            "sequence_length": DashboardBuildConfig().sequence_length,
        },
    }


def _descending_rank_score(rank_series: pd.Series) -> pd.Series:
    if len(rank_series) <= 1:
        return pd.Series(np.ones(len(rank_series)), index=rank_series.index, dtype=float)
    return 1.0 - (rank_series.astype(float) - 1.0) / float(len(rank_series) - 1)


def _percentile_rank(series: pd.Series, value: float) -> float:
    clean = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return float("nan")
    return float((clean <= value).mean())


def _parse_formula(formula: str) -> tuple[str, str, str]:
    parts = formula.split()
    if len(parts) != 3:
        raise ValueError(f"Unsupported formula format: {formula}")
    return parts[0], parts[1], parts[2]


def _operator_hint(operator: str) -> str:
    if operator == "-":
        return "A larger left-side signal and a smaller right-side signal increase the factor score."
    if operator == "+":
        return "Higher values on both inputs increase the factor score."
    if operator == "*":
        return "The score is amplified when both inputs are large in the same direction."
    if operator == "/":
        return "The score increases when the left-side signal is strong and the right-side signal stays contained."
    return "The RL factor combines the two signals into one ranking score."


def _rl_reason_text(
    formula: str,
    factor_value: float,
    rl_rank: int,
    factor_pool: list[dict[str, Any]],
    left_feature: str,
    left_value: float,
    left_percentile: float,
    right_feature: str,
    right_value: float,
    right_percentile: float,
    operator: str,
) -> str:
    if len(factor_pool) > 1:
        pool_preview = ", ".join(
            f"{item['formula']} ({float(item['weight']):+.3f})"
            for item in factor_pool[:3]
        )
        return (
            f"The RL module ranks this stock with a pooled factor score of {factor_value:.4f}, which places it at "
            f"RL rank #{rl_rank}. The current alpha pool combines {len(factor_pool)} formulas. "
            f"Representative weighted terms include: {pool_preview}."
        )

    left_label = FEATURE_LABELS.get(left_feature, left_feature)
    right_label = FEATURE_LABELS.get(right_feature, right_feature)
    return (
        f"The RL factor `{formula}` gives this stock a latest score of {factor_value:.4f}, which places it at "
        f"RL rank #{rl_rank}. {left_label} is {left_value:.4f} ({left_percentile:.0%} percentile) and "
        f"{right_label} is {right_value:.4f} ({right_percentile:.0%} percentile). {_operator_hint(operator)}"
    )


def _fit_metrics(stock_fit_frame: pd.DataFrame) -> dict[str, float]:
    if stock_fit_frame.empty:
        return {}
    actual = stock_fit_frame["target"].to_numpy(dtype=np.float64)
    predicted = stock_fit_frame["prediction"].to_numpy(dtype=np.float64)
    mse = float(np.mean((actual - predicted) ** 2))
    mae = float(np.mean(np.abs(actual - predicted)))
    return {
        "mse": mse,
        "mae": mae,
        "points": float(len(stock_fit_frame)),
    }


def _dashboard_split_config(panel: pd.DataFrame) -> SplitConfig:
    latest_labeled_dates = panel.loc[panel["future_return_20d"].notna(), "trade_date"]
    latest_labeled_date = int(latest_labeled_dates.max()) if not latest_labeled_dates.empty else int(panel["trade_date"].max())
    return SplitConfig(test_end=latest_labeled_date)


def _augment_rl_features(panel: pd.DataFrame) -> pd.DataFrame:
    working = panel.copy()
    grouped = working.sort_values(["ts_code", "trade_date"]).groupby("ts_code", group_keys=False)
    working["intraday_strength"] = working["close_adj"] / working["open_adj"] - 1.0
    working["ma_gap_5"] = working["close_adj"] / working["ma_5"] - 1.0
    working["ma_gap_10"] = working["close_adj"] / working["ma_10"] - 1.0
    working["trend_gap"] = working["ma_5"] / working["ma_10"] - 1.0
    working["return_10d"] = grouped["close_adj"].pct_change(10)
    working["volume_ratio_5"] = working["vol"] / grouped["vol"].transform(
        lambda series: series.rolling(5, min_periods=3).mean()
    )
    working["volume_ratio_10"] = working["vol"] / grouped["vol"].transform(
        lambda series: series.rolling(10, min_periods=5).mean()
    )
    return working.replace([np.inf, -np.inf], np.nan)

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import pandas as pd
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def mean_daily_rank_ic(
    frame: pd.DataFrame,
    prediction_col: str,
    target_col: str,
    date_col: str = "trade_date",
) -> float:
    working = frame[[date_col, prediction_col, target_col]].replace([np.inf, -np.inf], np.nan)
    working = working.dropna(subset=[prediction_col, target_col])
    if working.empty:
        return float("nan")

    daily_values: list[float] = []
    for _, group in working.groupby(date_col):
        if len(group) < 2:
            continue
        if group[prediction_col].nunique(dropna=True) < 2:
            continue
        if group[target_col].nunique(dropna=True) < 2:
            continue
        corr = group[prediction_col].corr(group[target_col], method="spearman")
        if pd.notna(corr):
            daily_values.append(float(corr))

    if not daily_values:
        return float("nan")
    return float(np.mean(daily_values))


def pearson_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    return float(corr)


def rank_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    series_true = pd.Series(y_true)
    series_pred = pd.Series(y_pred)
    if series_true.nunique(dropna=True) < 2:
        return float("nan")
    if series_pred.nunique(dropna=True) < 2:
        return float("nan")
    corr = series_true.corr(series_pred, method="spearman")
    return float(corr)


def regression_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(math.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "ic": pearson_ic(y_true, y_pred),
        "rank_ic": rank_ic(y_true, y_pred),
    }


def clean_number(value: Any) -> float:
    if value is None:
        return float("nan")
    return float(value)

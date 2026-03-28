from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from integrated_alpha.common.config import SplitConfig
from integrated_alpha.common.io_utils import ensure_directory, save_json
from integrated_alpha.common.metrics import mean_daily_rank_ic, regression_summary, seed_everything
from integrated_alpha.data_module.panel_data import FEATURE_COLUMNS, TARGET_COLUMN


@dataclass(frozen=True)
class LSTMConfig:
    sequence_length: int = 20
    hidden_size: int = 32
    batch_size: int = 128
    epochs: int = 5
    learning_rate: float = 1e-3
    patience: int = 2
    stock_limit: int = 20
    random_seed: int = 42


@dataclass(frozen=True)
class PriceDemoConfig:
    stock_code: str = "000001.SZ"
    sequence_length: int = 30
    hidden_size: int = 64
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 1e-3
    patience: int = 5
    random_seed: int = 42


class SequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: np.ndarray) -> None:
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.sequences.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[index], self.targets[index]


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(inputs)
        last_hidden = outputs[:, -1, :]
        prediction = self.head(last_hidden).squeeze(-1)
        return prediction


def run_lstm_pipeline(
    panel: pd.DataFrame,
    output_dir: Path,
    split_config: SplitConfig,
    config: LSTMConfig,
) -> dict[str, Any]:
    ensure_directory(output_dir)
    seed_everything(config.random_seed)

    scaler = _fit_scaler(panel=panel, split_config=split_config)
    train_pack, val_pack, test_pack = _build_sequence_packs(
        panel=panel,
        split_config=split_config,
        scaler=scaler,
        sequence_length=config.sequence_length,
    )

    train_dataset = SequenceDataset(train_pack["x"], train_pack["y"])
    val_dataset = SequenceDataset(val_pack["x"], val_pack["y"])
    test_dataset = SequenceDataset(test_pack["x"], test_pack["y"])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    model = LSTMRegressor(input_size=len(FEATURE_COLUMNS), hidden_size=config.hidden_size)
    history = _train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        patience=config.patience,
    )

    val_predictions = _predict(model, val_loader)
    predictions = _predict(model, test_loader)
    val_target_values = val_pack["y"].astype(np.float64)
    target_values = test_pack["y"].astype(np.float64)
    val_prediction_frame = val_pack["meta"].copy()
    val_prediction_frame["prediction"] = val_predictions
    val_prediction_frame["target"] = val_target_values
    prediction_frame = test_pack["meta"].copy()
    prediction_frame["prediction"] = predictions
    prediction_frame["target"] = target_values

    summary = regression_summary(target_values, predictions)
    summary["mean_daily_rank_ic"] = mean_daily_rank_ic(
        prediction_frame,
        prediction_col="prediction",
        target_col="target",
    )
    summary["baseline_mean_daily_rank_ic"] = mean_daily_rank_ic(
        prediction_frame,
        prediction_col="baseline_return_5d",
        target_col="target",
    )
    summary["num_stocks"] = int(panel["ts_code"].nunique())
    summary["train_sequences"] = int(len(train_pack["x"]))
    summary["val_sequences"] = int(len(val_pack["x"]))
    summary["test_sequences"] = int(len(test_pack["x"]))
    summary["val_mean_daily_rank_ic"] = mean_daily_rank_ic(
        val_prediction_frame,
        prediction_col="prediction",
        target_col="target",
    )

    val_prediction_frame.to_csv(output_dir / "val_predictions.csv", index=False, encoding="utf-8-sig")
    prediction_frame.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    save_json(summary, output_dir / "summary.json")
    _plot_losses(history, output_dir / "loss_curve.png")
    _plot_daily_prediction_comparison(
        prediction_frame=prediction_frame,
        output_path=output_dir / "actual_vs_predicted_daily.png",
    )
    _plot_representative_stock_comparison(
        prediction_frame=prediction_frame,
        output_path=output_dir / "actual_vs_predicted_stock.png",
        csv_output_path=output_dir / "representative_stock_series.csv",
    )

    return {
        "summary": summary,
        "val_prediction_frame": val_prediction_frame,
        "prediction_frame": prediction_frame,
        "history": history,
        "model": model,
        "scaler": scaler,
        "sequence_length": config.sequence_length,
    }


def run_price_demo_pipeline(
    panel: pd.DataFrame,
    output_dir: Path,
    split_config: SplitConfig,
    config: PriceDemoConfig,
) -> dict[str, Any]:
    ensure_directory(output_dir)
    seed_everything(config.random_seed)

    stock_frame = (
        panel.loc[panel["ts_code"] == config.stock_code]
        .sort_values("trade_date")
        .reset_index(drop=True)
        .copy()
    )
    if stock_frame.empty:
        raise ValueError(f"Stock {config.stock_code} was not found in the local panel.")

    feature_scaler, target_scaler = _fit_price_demo_scalers(stock_frame, split_config)
    train_pack, val_pack, test_pack = _build_price_demo_packs(
        stock_frame=stock_frame,
        split_config=split_config,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        sequence_length=config.sequence_length,
    )

    train_dataset = SequenceDataset(train_pack["x"], train_pack["y"])
    val_dataset = SequenceDataset(val_pack["x"], val_pack["y"])
    test_dataset = SequenceDataset(test_pack["x"], test_pack["y"])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    model = LSTMRegressor(input_size=len(FEATURE_COLUMNS), hidden_size=config.hidden_size)
    history = _train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        patience=config.patience,
    )

    predicted_scaled = _predict(model, test_loader).reshape(-1, 1)
    actual_scaled = test_pack["y"].astype(np.float64).reshape(-1, 1)
    predicted_price = target_scaler.inverse_transform(predicted_scaled).reshape(-1)
    actual_price = target_scaler.inverse_transform(actual_scaled).reshape(-1)

    result_frame = test_pack["meta"].copy()
    result_frame["actual_price"] = actual_price
    result_frame["predicted_price"] = predicted_price
    result_frame["trade_date_dt"] = pd.to_datetime(result_frame["trade_date"].astype(str), format="%Y%m%d")

    summary = _price_demo_summary(actual_price, predicted_price)
    summary["stock_code"] = config.stock_code
    summary["train_sequences"] = int(len(train_pack["x"]))
    summary["val_sequences"] = int(len(val_pack["x"]))
    summary["test_sequences"] = int(len(test_pack["x"]))

    result_frame.to_csv(output_dir / "price_demo_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history).to_csv(output_dir / "price_demo_training_history.csv", index=False, encoding="utf-8-sig")
    save_json(summary, output_dir / "price_demo_summary.json")
    _plot_losses(history, output_dir / "price_demo_loss_curve.png")
    _plot_price_demo(result_frame, output_dir / "price_demo_actual_vs_predicted.png")

    return {"summary": summary, "prediction_frame": result_frame, "history": history}


def _fit_scaler(panel: pd.DataFrame, split_config: SplitConfig) -> StandardScaler:
    train_rows = panel.loc[panel["trade_date"] <= split_config.train_end].copy()
    train_rows = train_rows.dropna(subset=FEATURE_COLUMNS)
    scaler = StandardScaler()
    scaler.fit(train_rows[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
    return scaler


def _fit_price_demo_scalers(
    stock_frame: pd.DataFrame,
    split_config: SplitConfig,
) -> tuple[StandardScaler, StandardScaler]:
    train_rows = stock_frame.loc[stock_frame["trade_date"] <= split_config.train_end].copy()
    train_rows = train_rows.dropna(subset=FEATURE_COLUMNS + ["close_adj"])
    feature_scaler = StandardScaler()
    feature_scaler.fit(train_rows[FEATURE_COLUMNS].to_numpy(dtype=np.float32))

    target_scaler = StandardScaler()
    target_scaler.fit(train_rows[["close_adj"]].to_numpy(dtype=np.float32))
    return feature_scaler, target_scaler


def _build_sequence_packs(
    panel: pd.DataFrame,
    split_config: SplitConfig,
    scaler: StandardScaler,
    sequence_length: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    split_to_sequences: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    split_to_targets: dict[str, list[float]] = {"train": [], "val": [], "test": []}
    split_to_meta: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}

    working = panel.copy()
    working = working.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN, "return_5d"])
    working.loc[:, FEATURE_COLUMNS] = scaler.transform(working[FEATURE_COLUMNS].to_numpy(dtype=np.float32))

    for ts_code, group in working.groupby("ts_code"):
        group = group.sort_values("trade_date").reset_index(drop=True)
        features = group[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        targets = group[TARGET_COLUMN].to_numpy(dtype=np.float32)
        trade_dates = group["trade_date"].to_numpy(dtype=np.int64)
        baseline_return_5d = group["return_5d"].to_numpy(dtype=np.float32)

        if len(group) <= sequence_length:
            continue

        for index in range(sequence_length, len(group)):
            trade_date = int(trade_dates[index])
            split_name = _assign_split_name(trade_date, split_config)
            if split_name is None:
                continue

            split_to_sequences[split_name].append(features[index - sequence_length : index])
            split_to_targets[split_name].append(float(targets[index]))
            split_to_meta[split_name].append(
                {
                    "ts_code": ts_code,
                    "trade_date": trade_date,
                    "baseline_return_5d": float(baseline_return_5d[index]),
                }
            )

    return tuple(
        {
            "x": np.asarray(split_to_sequences[name], dtype=np.float32),
            "y": np.asarray(split_to_targets[name], dtype=np.float32),
            "meta": pd.DataFrame(split_to_meta[name]),
        }
        for name in ("train", "val", "test")
    )


def _build_price_demo_packs(
    stock_frame: pd.DataFrame,
    split_config: SplitConfig,
    feature_scaler: StandardScaler,
    target_scaler: StandardScaler,
    sequence_length: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    split_to_sequences: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    split_to_targets: dict[str, list[float]] = {"train": [], "val": [], "test": []}
    split_to_meta: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}

    working = stock_frame.copy()
    working = working.dropna(subset=FEATURE_COLUMNS + ["close_adj"]).reset_index(drop=True)
    raw_close = working["close_adj"].to_numpy(dtype=np.float32).reshape(-1, 1)
    scaled_features = feature_scaler.transform(working[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
    target_scaled = target_scaler.transform(raw_close).reshape(-1)
    working.loc[:, FEATURE_COLUMNS] = scaled_features
    working.loc[:, "target_scaled"] = target_scaled

    features = working[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    targets = working["target_scaled"].to_numpy(dtype=np.float32)
    trade_dates = working["trade_date"].to_numpy(dtype=np.int64)

    for index in range(sequence_length, len(working)):
        trade_date = int(trade_dates[index])
        split_name = _assign_split_name(trade_date, split_config)
        if split_name is None:
            continue

        split_to_sequences[split_name].append(features[index - sequence_length : index])
        split_to_targets[split_name].append(float(targets[index]))
        split_to_meta[split_name].append(
            {
                "ts_code": str(working.loc[index, "ts_code"]),
                "trade_date": trade_date,
            }
        )

    return tuple(
        {
            "x": np.asarray(split_to_sequences[name], dtype=np.float32),
            "y": np.asarray(split_to_targets[name], dtype=np.float32),
            "meta": pd.DataFrame(split_to_meta[name]),
        }
        for name in ("train", "val", "test")
    )


def _assign_split_name(trade_date: int, split_config: SplitConfig) -> str | None:
    if trade_date <= split_config.train_end:
        return "train"
    if trade_date <= split_config.val_end:
        return "val"
    if trade_date <= split_config.test_end:
        return "test"
    return None


def _train_model(
    model: LSTMRegressor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    patience: int,
) -> list[dict[str, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            prediction = model(batch_x)
            loss = loss_fn(prediction, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                prediction = model(batch_x)
                loss = loss_fn(prediction, batch_y)
                val_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            break

    model.load_state_dict(best_state)
    return history


def _predict(model: LSTMRegressor, data_loader: DataLoader) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for batch_x, _ in data_loader:
            prediction = model(batch_x).cpu().numpy()
            outputs.append(prediction)
    if not outputs:
        return np.array([], dtype=np.float32)
    return np.concatenate(outputs).astype(np.float64)


def predict_latest_returns(
    panel: pd.DataFrame,
    model: LSTMRegressor,
    scaler: StandardScaler,
    sequence_length: int,
) -> pd.DataFrame:
    working = panel.copy()
    working = working.dropna(subset=FEATURE_COLUMNS)
    if working.empty:
        return pd.DataFrame(columns=["ts_code", "trade_date", "predicted_future_return_20d"])

    working.loc[:, FEATURE_COLUMNS] = scaler.transform(working[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
    rows: list[dict[str, Any]] = []

    for ts_code, group in working.groupby("ts_code"):
        group = group.sort_values("trade_date").reset_index(drop=True)
        if len(group) <= sequence_length:
            continue

        features = group[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        prediction_date = int(group.loc[len(group) - 1, "trade_date"])
        latest_sequence = features[-sequence_length - 1 : -1]
        if latest_sequence.shape[0] != sequence_length:
            continue

        sequence_tensor = torch.tensor(latest_sequence[None, :, :], dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            prediction = float(model(sequence_tensor).cpu().numpy().reshape(-1)[0])

        rows.append(
            {
                "ts_code": str(ts_code),
                "trade_date": prediction_date,
                "predicted_future_return_20d": prediction,
            }
        )

    return pd.DataFrame(rows)


def _plot_losses(history: list[dict[str, float]], output_path: Path) -> None:
    history_frame = pd.DataFrame(history)
    plt.figure(figsize=(10, 4))
    plt.plot(history_frame["epoch"], history_frame["train_loss"], label="train")
    plt.plot(history_frame["epoch"], history_frame["val_loss"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("LSTM loss curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_daily_prediction_comparison(prediction_frame: pd.DataFrame, output_path: Path) -> None:
    if prediction_frame.empty:
        return

    daily = (
        prediction_frame.groupby("trade_date", as_index=False)[["prediction", "target"]]
        .mean()
        .sort_values("trade_date")
        .reset_index(drop=True)
    )
    daily["trade_date_dt"] = pd.to_datetime(daily["trade_date"].astype(str), format="%Y%m%d")

    plt.figure(figsize=(11, 4))
    plt.plot(daily["trade_date_dt"], daily["target"], label="actual future_return_20d", linewidth=1.8)
    plt.plot(daily["trade_date_dt"], daily["prediction"], label="predicted future_return_20d", linewidth=1.8)
    plt.xlabel("Trade date")
    plt.ylabel("Return")
    plt.title("Daily Mean Actual vs Predicted Return")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_representative_stock_comparison(
    prediction_frame: pd.DataFrame,
    output_path: Path,
    csv_output_path: Path,
) -> None:
    if prediction_frame.empty:
        return

    stock_counts = prediction_frame["ts_code"].value_counts()
    representative_stock = str(stock_counts.index[0])
    stock_frame = (
        prediction_frame.loc[prediction_frame["ts_code"] == representative_stock]
        .sort_values("trade_date")
        .reset_index(drop=True)
        .copy()
    )
    stock_frame["trade_date_dt"] = pd.to_datetime(stock_frame["trade_date"].astype(str), format="%Y%m%d")
    stock_frame.to_csv(csv_output_path, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(11, 4))
    plt.plot(stock_frame["trade_date_dt"], stock_frame["target"], label="actual future_return_20d", linewidth=1.8)
    plt.plot(
        stock_frame["trade_date_dt"],
        stock_frame["prediction"],
        label="predicted future_return_20d",
        linewidth=1.8,
    )
    plt.xlabel("Trade date")
    plt.ylabel("Return")
    plt.title(f"Actual vs Predicted Return for {representative_stock}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _price_demo_summary(actual_price: np.ndarray, predicted_price: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual_price, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted_price, dtype=np.float64).reshape(-1)
    mse = float(np.mean((actual - predicted) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(actual - predicted)))
    denominator = np.where(np.abs(actual) < 1e-8, np.nan, np.abs(actual))
    mape = float(np.nanmean(np.abs((actual - predicted) / denominator)) * 100.0)
    return {"mse": mse, "rmse": rmse, "mae": mae, "mape_percent": mape}


def _plot_price_demo(prediction_frame: pd.DataFrame, output_path: Path) -> None:
    if prediction_frame.empty:
        return

    ordered = prediction_frame.sort_values("trade_date_dt").reset_index(drop=True)
    plt.figure(figsize=(12, 6))
    plt.plot(ordered["trade_date_dt"], ordered["actual_price"], label="Actual Price", linewidth=1.8)
    plt.plot(ordered["trade_date_dt"], ordered["predicted_price"], label="Predicted Price", linewidth=1.8)
    plt.title(f"LSTM Stock Price Prediction: {ordered['ts_code'].iloc[0]}")
    plt.xlabel("Trade date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

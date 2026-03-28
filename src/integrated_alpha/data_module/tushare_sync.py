from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import os
import pandas as pd
import tushare as ts
from dotenv import load_dotenv

from integrated_alpha.common.io_utils import ensure_directory, save_json


@dataclass(frozen=True)
class TushareSyncConfig:
    start_date: int = 20180101
    pause_seconds: float = 1.30
    limit_stocks: int | None = None
    incremental: bool = True
    lookback_days: int = 90
    max_retries: int = 3
    retry_wait_seconds: float = 65.0
    transient_wait_seconds: float = 10.0


class TushareDailySync:
    def __init__(self, token: str) -> None:
        if not token.strip():
            raise RuntimeError("TUSHARE_TOKEN is missing. Add it to .env before running tushare sync.")
        clean_token = token.strip()
        os.environ["TUSHARE_TOKEN"] = clean_token
        self.pro = ts.pro_api(clean_token)

    @classmethod
    def from_env(cls, project_root: Path) -> "TushareDailySync":
        load_dotenv(project_root / ".env")
        token = os.getenv("TUSHARE_TOKEN", "").strip()
        return cls(token=token)

    def rebuild_existing_universe(
        self,
        source_dir: Path,
        destination_dir: Path,
        config: TushareSyncConfig,
    ) -> dict[str, Any]:
        ensure_directory(destination_dir)
        csv_paths = sorted(source_dir.glob("*.csv"))
        if not csv_paths:
            raise FileNotFoundError(f"No source CSV files found in {source_dir}")

        if config.limit_stocks is not None:
            csv_paths = csv_paths[: config.limit_stocks]

        target_latest_date = _infer_directory_latest_trade_date(destination_dir) or _infer_directory_latest_trade_date(source_dir)
        summary_rows: list[dict[str, Any]] = []
        for index, csv_path in enumerate(csv_paths, start=1):
            ts_code = csv_path.stem
            existing_destination = destination_dir / f"{ts_code}.csv"
            sync_start_date = config.start_date
            mode = "full"
            existing_rows = 0

            if config.incremental and existing_destination.exists():
                existing_frame = pd.read_csv(existing_destination)
                existing_rows = int(len(existing_frame))
                if not existing_frame.empty and "trade_date" in existing_frame.columns:
                    existing_latest_trade_date = int(existing_frame["trade_date"].max())
                    if target_latest_date is not None and existing_latest_trade_date >= target_latest_date:
                        summary_rows.append(
                            {
                                "ts_code": ts_code,
                                "status": "skipped",
                                "rows": existing_rows,
                                "mode": "incremental",
                                "date_end": existing_latest_trade_date,
                            }
                        )
                        continue
                    latest_date = pd.to_datetime(str(existing_latest_trade_date), format="%Y%m%d")
                    sync_start_date = int((latest_date - pd.Timedelta(days=config.lookback_days)).strftime("%Y%m%d"))
                    mode = "incremental"

            history = self._fetch_daily_with_retry(
                ts_code=ts_code,
                start_date=sync_start_date,
                config=config,
            )
            if history is None or history.empty:
                summary_rows.append(
                    {
                        "ts_code": ts_code,
                        "status": "empty",
                        "rows": 0,
                        "mode": mode,
                    }
                )
                time.sleep(config.pause_seconds)
                continue

            labeled = build_labeled_frame(history)
            if mode == "incremental" and existing_destination.exists():
                merged = _merge_incremental_frame(existing_destination=existing_destination, refreshed_frame=labeled)
                merged.to_csv(existing_destination, index=False, encoding="utf-8-sig")
                final_frame = merged
            else:
                labeled.to_csv(existing_destination, index=False, encoding="utf-8-sig")
                final_frame = labeled
            summary_rows.append(
                {
                    "ts_code": ts_code,
                    "status": "ok",
                    "mode": mode,
                    "rows": int(len(final_frame)),
                    "new_rows": int(max(len(final_frame) - existing_rows, 0)),
                    "date_start": int(final_frame["trade_date"].min()),
                    "date_end": int(final_frame["trade_date"].max()),
                    "sync_start_date": int(sync_start_date),
                }
            )
            time.sleep(config.pause_seconds)

        summary = {
            "source_dir": str(source_dir),
            "destination_dir": str(destination_dir),
            "requested_stocks": int(len(csv_paths)),
            "completed_stocks": int(sum(row["status"] == "ok" for row in summary_rows)),
            "empty_stocks": int(sum(row["status"] == "empty" for row in summary_rows)),
            "rows": summary_rows,
        }
        return summary

    def _fetch_daily_with_retry(
        self,
        ts_code: str,
        start_date: int,
        config: TushareSyncConfig,
    ) -> pd.DataFrame:
        for attempt in range(1, config.max_retries + 1):
            try:
                return self.pro.daily(
                    ts_code=ts_code,
                    start_date=str(start_date),
                )
            except Exception as exc:
                message = str(exc)
                if "每分钟最多访问该接口50次" in message and attempt < config.max_retries:
                    time.sleep(config.retry_wait_seconds)
                    continue
                if attempt < config.max_retries:
                    time.sleep(config.transient_wait_seconds)
                    continue
                raise


def build_labeled_frame(history: pd.DataFrame) -> pd.DataFrame:
    required = ["ts_code", "trade_date", "open", "high", "low", "close", "vol"]
    missing = [column for column in required if column not in history.columns]
    if missing:
        raise ValueError(f"Tushare daily result is missing required columns: {missing}")

    frame = history[required].copy()
    frame = frame.sort_values("trade_date").reset_index(drop=True)
    frame["trade_date"] = frame["trade_date"].astype(int)

    frame = frame.rename(
        columns={
            "open": "open_adj",
            "high": "high_adj",
            "low": "low_adj",
            "close": "close_adj",
            "vol": "vol",
        }
    )

    close = frame["close_adj"].astype(float)
    frame["return_1d"] = close.pct_change()
    frame["return_5d"] = close / close.shift(5) - 1.0
    frame["ma_5"] = close.rolling(window=5).mean()
    frame["ma_10"] = close.rolling(window=10).mean()
    frame["volatility_5"] = frame["return_1d"].rolling(window=5).std()
    frame["future_return_1d"] = close.shift(-1) / close - 1.0
    frame["future_return_5d"] = close.shift(-5) / close - 1.0
    frame["future_return_20d"] = close.shift(-20) / close - 1.0

    return frame[
        [
            "ts_code",
            "trade_date",
            "open_adj",
            "high_adj",
            "low_adj",
            "close_adj",
            "vol",
            "return_1d",
            "return_5d",
            "ma_5",
            "ma_10",
            "volatility_5",
            "future_return_1d",
            "future_return_5d",
            "future_return_20d",
        ]
    ]


def _merge_incremental_frame(existing_destination: Path, refreshed_frame: pd.DataFrame) -> pd.DataFrame:
    existing = pd.read_csv(existing_destination)
    combined = pd.concat([existing, refreshed_frame], ignore_index=True)
    combined["trade_date"] = combined["trade_date"].astype(int)
    combined = combined.sort_values(["trade_date"]).drop_duplicates(subset=["trade_date"], keep="last")
    return combined.reset_index(drop=True)


def _infer_directory_latest_trade_date(directory: Path) -> int | None:
    latest: int | None = None
    for path in directory.glob("*.csv"):
        try:
            frame = pd.read_csv(path, usecols=["trade_date"])
        except Exception:
            continue
        if frame.empty:
            continue
        current = int(frame["trade_date"].max())
        latest = current if latest is None else max(latest, current)
    return latest


def run_tushare_sync(
    project_root: Path,
    config: TushareSyncConfig | None = None,
) -> dict[str, Any]:
    config = config or TushareSyncConfig()
    project_root = Path(project_root)
    source_dir = project_root / "data" / "labeled"
    destination_dir = project_root / "data" / "tushare_labeled"
    outputs_dir = project_root / "outputs" / "data_sync"

    syncer = TushareDailySync.from_env(project_root=project_root)
    summary = syncer.rebuild_existing_universe(
        source_dir=source_dir,
        destination_dir=destination_dir,
        config=config,
    )
    ensure_directory(outputs_dir)
    save_json(summary, outputs_dir / "tushare_sync_summary.json")
    return summary

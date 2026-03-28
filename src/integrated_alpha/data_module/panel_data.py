from __future__ import annotations

from pathlib import Path

import pandas as pd

from integrated_alpha.common.config import SplitConfig
from integrated_alpha.common.io_utils import ensure_directory

FEATURE_COLUMNS = [
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
]
TARGET_COLUMN = "future_return_20d"
REQUIRED_COLUMNS = ["ts_code", "trade_date", *FEATURE_COLUMNS, TARGET_COLUMN]


class PanelDataManager:
    def __init__(self, data_dir: Path, cache_dir: Path) -> None:
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        ensure_directory(self.cache_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        self.csv_paths = sorted(self.data_dir.glob("*.csv"))
        if not self.csv_paths:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")

    def load_panel(
        self,
        stock_codes: list[str] | None = None,
        force_reload: bool = False,
    ) -> pd.DataFrame:
        if stock_codes is None:
            cache_path = self.cache_dir / "panel_full.pkl"
            if cache_path.exists() and not force_reload:
                return pd.read_pickle(cache_path)

        selected_paths = self._select_paths(stock_codes)
        frames: list[pd.DataFrame] = []
        for csv_path in selected_paths:
            frame = pd.read_csv(csv_path, usecols=REQUIRED_COLUMNS)
            frames.append(frame)

        panel = pd.concat(frames, ignore_index=True)
        panel["trade_date"] = panel["trade_date"].astype(int)
        panel = panel.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)

        if stock_codes is None:
            panel.to_pickle(self.cache_dir / "panel_full.pkl")
        return panel

    def summarize(self, panel: pd.DataFrame) -> dict[str, int]:
        latest_trade_date = int(panel["trade_date"].max())
        latest_stock_count = int((panel["trade_date"] == latest_trade_date).sum())
        return {
            "stock_count": int(panel["ts_code"].nunique()),
            "row_count": int(len(panel)),
            "date_start": int(panel["trade_date"].min()),
            "date_end": int(panel["trade_date"].max()),
            "latest_trade_date": latest_trade_date,
            "latest_stock_count": latest_stock_count,
        }

    def split_by_date(
        self,
        panel: pd.DataFrame,
        split_config: SplitConfig,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        trade_date = panel["trade_date"]
        train = panel.loc[trade_date <= split_config.train_end].copy()
        val = panel.loc[(trade_date > split_config.train_end) & (trade_date <= split_config.val_end)].copy()
        test = panel.loc[(trade_date > split_config.val_end) & (trade_date <= split_config.test_end)].copy()
        return train, val, test

    def filter_stocks(self, panel: pd.DataFrame, stock_codes: list[str]) -> pd.DataFrame:
        return panel.loc[panel["ts_code"].isin(stock_codes)].copy()

    def latest_snapshot(self, panel: pd.DataFrame) -> pd.DataFrame:
        latest_trade_date = int(panel["trade_date"].max())
        return panel.loc[panel["trade_date"] == latest_trade_date].copy()

    def select_evenly_spaced_stock_codes(self, limit: int) -> list[str]:
        all_codes = [path.stem for path in self.csv_paths]
        if limit >= len(all_codes):
            return all_codes

        step = max(len(all_codes) // limit, 1)
        selected = all_codes[::step][:limit]
        return selected

    def _select_paths(self, stock_codes: list[str] | None) -> list[Path]:
        if stock_codes is None:
            return self.csv_paths

        wanted = {f"{code}.csv" if not code.endswith(".csv") else code for code in stock_codes}
        selected_paths = [path for path in self.csv_paths if path.name in wanted]
        if not selected_paths:
            raise ValueError("No requested stock codes were found in the local dataset.")
        return selected_paths

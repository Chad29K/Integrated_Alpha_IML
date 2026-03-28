from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from integrated_alpha.data_module.tushare_sync import build_labeled_frame


def main() -> None:
    frame = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"] * 25,
            "trade_date": [20240101 + index for index in range(25)],
            "open": [10.0 + index for index in range(25)],
            "high": [10.5 + index for index in range(25)],
            "low": [9.5 + index for index in range(25)],
            "close": [10.2 + index for index in range(25)],
            "vol": [1000 + 10 * index for index in range(25)],
        }
    )
    labeled = build_labeled_frame(frame)
    assert "future_return_20d" in labeled.columns
    assert len(labeled) == 25
    assert labeled["close_adj"].iloc[0] == 10.2
    assert pd.notna(labeled["future_return_20d"].iloc[0])
    print("Tushare sync smoke test passed.")


if __name__ == "__main__":
    main()

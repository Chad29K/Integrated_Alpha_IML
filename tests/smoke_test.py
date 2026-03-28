from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from integrated_alpha.common.config import ProjectPaths, SplitConfig
from integrated_alpha.data_module.panel_data import PanelDataManager
from integrated_alpha.lstm_module.model import LSTMConfig, run_lstm_pipeline
from integrated_alpha.rl_module.symbolic_factor_agent import RLSearchConfig, run_rl_pipeline


def main() -> None:
    paths = ProjectPaths.from_root(ROOT)
    manager = PanelDataManager(paths.data_dir, paths.cache_dir)
    stock_codes = manager.select_evenly_spaced_stock_codes(limit=12)
    panel = manager.load_panel(stock_codes=stock_codes, force_reload=True)

    split_config = SplitConfig(train_end=20221230, val_end=20241231, test_end=20251231)

    rl_result = run_rl_pipeline(
        panel=panel,
        output_dir=paths.rl_dir / "smoke",
        split_config=split_config,
        config=RLSearchConfig(episodes=8, random_seed=7, top_n=5),
    )
    assert rl_result["summary"]["best_formula"]

    lstm_result = run_lstm_pipeline(
        panel=panel,
        output_dir=paths.lstm_dir / "smoke",
        split_config=split_config,
        config=LSTMConfig(epochs=1, stock_limit=12, random_seed=7),
    )
    assert lstm_result["summary"]["num_stocks"] > 0

    print("Smoke test passed.")


if __name__ == "__main__":
    main()

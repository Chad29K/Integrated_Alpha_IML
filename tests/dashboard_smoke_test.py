from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from integrated_alpha.dashboard_module.service import explain_stock_pick, get_dashboard_bundle


def main() -> None:
    bundle = get_dashboard_bundle(ROOT)
    assert not bundle["top_recommendations"].empty

    stock_code = str(bundle["top_recommendations"].iloc[0]["ts_code"])
    detail = explain_stock_pick(bundle, stock_code)
    assert detail["stock_code"] == stock_code
    assert detail["reasons"]
    assert "formula" in detail

    print("Dashboard smoke test passed.")


if __name__ == "__main__":
    main()

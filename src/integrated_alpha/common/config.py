from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SplitConfig:
    train_end: int = 20221230
    val_end: int = 20241231
    test_end: int = 20251231


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    data_dir: Path
    fallback_data_dir: Path
    tushare_data_dir: Path
    output_dir: Path
    cache_dir: Path
    rl_dir: Path
    lstm_dir: Path
    llm_dir: Path
    combined_dir: Path
    dashboard_dir: Path

    @classmethod
    def from_root(cls, project_root: Path) -> "ProjectPaths":
        output_dir = project_root / "outputs"
        fallback_data_dir = project_root / "data" / "labeled"
        tushare_data_dir = project_root / "data" / "tushare_labeled"
        fallback_count = len(list(fallback_data_dir.glob("*.csv")))
        tushare_count = len(list(tushare_data_dir.glob("*.csv")))
        use_tushare = fallback_count > 0 and tushare_count >= fallback_count
        data_dir = tushare_data_dir if use_tushare else fallback_data_dir
        return cls(
            project_root=project_root,
            data_dir=data_dir,
            fallback_data_dir=fallback_data_dir,
            tushare_data_dir=tushare_data_dir,
            output_dir=output_dir,
            cache_dir=output_dir / "cache",
            rl_dir=output_dir / "rl",
            lstm_dir=output_dir / "lstm",
            llm_dir=output_dir / "llm",
            combined_dir=output_dir / "combined",
            dashboard_dir=output_dir / "dashboard",
        )

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from integrated_alpha.common.config import SplitConfig
from integrated_alpha.common.io_utils import ensure_directory, save_json
from integrated_alpha.common.metrics import mean_daily_rank_ic
from integrated_alpha.data_module.panel_data import TARGET_COLUMN

RL_FEATURE_COLUMNS = (
    "return_1d",
    "return_5d",
    "return_10d",
    "volatility_5",
    "intraday_strength",
    "ma_gap_5",
    "ma_gap_10",
    "trend_gap",
    "volume_ratio_5",
    "volume_ratio_10",
)


@dataclass(frozen=True)
class RLSearchConfig:
    episodes: int = 120
    alpha: float = 0.35
    gamma: float = 0.90
    epsilon_start: float = 1.0
    epsilon_decay: float = 0.985
    epsilon_min: float = 0.05
    top_n: int = 10
    random_seed: int = 42
    features: tuple[str, ...] = RL_FEATURE_COLUMNS
    operators: tuple[str, ...] = ("+", "-", "*", "/")
    train_reward_weight: float = 0.30
    validation_reward_weight: float = 1.00
    stability_penalty_weight: float = 1.00
    pool_size: int = 4
    max_formula_correlation: float = 0.80


class SymbolicFactorRLAgent:
    def __init__(self, config: RLSearchConfig) -> None:
        self.config = config
        self.random = random.Random(config.random_seed)
        self.q_table: dict[str, dict[str, float]] = {}

    def train(
        self,
        scorer: "FormulaScorer",
    ) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
        history_rows: list[dict[str, Any]] = []
        epsilon = self.config.epsilon_start
        current_pool: list[str] = []
        best_pool_summary: dict[str, Any] | None = None

        for episode in range(1, self.config.episodes + 1):
            formula, trajectory = self._sample_formula(epsilon)
            stats = scorer.score_formula(formula)
            pool_update = scorer.evaluate_formula_in_pool(formula=formula, current_pool=current_pool)
            reward = pool_update["reward"]
            self._update_q_values(trajectory, reward)

            current_pool = list(pool_update["updated_pool"])
            if best_pool_summary is None or pool_update["updated_summary"]["selection_score"] > best_pool_summary["selection_score"]:
                best_pool_summary = dict(pool_update["updated_summary"])

            history_rows.append(
                {
                    "episode": episode,
                    "epsilon": epsilon,
                    "formula": formula,
                    "reward": reward,
                    "selection_score": stats["selection_score"],
                    "train_rank_ic": stats["train_rank_ic"],
                    "val_rank_ic": stats["val_rank_ic"],
                    "test_rank_ic": stats["test_rank_ic"],
                    "pool_size": len(current_pool),
                    "pool_selection_score": pool_update["updated_summary"]["selection_score"],
                    "pool_val_rank_ic": pool_update["updated_summary"]["val_rank_ic"],
                    "selected_best_formula": str(best_pool_summary["best_formula"]) if best_pool_summary else formula,
                    "selected_best_score": float(best_pool_summary["selection_score"]) if best_pool_summary else reward,
                }
            )
            epsilon = max(self.config.epsilon_min, epsilon * self.config.epsilon_decay)

        history = pd.DataFrame(history_rows)
        all_scores = scorer.formula_score_table()
        summary = best_pool_summary or scorer.build_summary(best_formula=None, pool_formulas=current_pool)
        return history, summary, all_scores

    def _sample_formula(self, epsilon: float) -> tuple[str, list[tuple[str, str, str | None]]]:
        trajectory: list[tuple[str, str, str | None]] = []

        state = "START"
        left = self._choose_action(state, list(self.config.features), epsilon)
        next_state = f"LEFT::{left}"
        trajectory.append((state, left, next_state))

        operator = self._choose_action(next_state, list(self.config.operators), epsilon)
        next_state_2 = f"LEFT_OP::{left}::{operator}"
        trajectory.append((next_state, operator, next_state_2))

        right = self._choose_action(next_state_2, list(self.config.features), epsilon)
        trajectory.append((next_state_2, right, None))

        formula = f"{left} {operator} {right}"
        return formula, trajectory

    def _choose_action(self, state: str, actions: list[str], epsilon: float) -> str:
        q_values = self.q_table.setdefault(state, {action: 0.0 for action in actions})
        for action in actions:
            q_values.setdefault(action, 0.0)

        if self.random.random() < epsilon:
            return self.random.choice(actions)

        best_value = max(q_values[action] for action in actions)
        best_actions = [action for action in actions if q_values[action] == best_value]
        return self.random.choice(best_actions)

    def _update_q_values(self, trajectory: list[tuple[str, str, str | None]], terminal_reward: float) -> None:
        next_max = 0.0
        reward = terminal_reward

        for state, action, _ in reversed(trajectory):
            state_q = self.q_table.setdefault(state, {})
            current_q = state_q.get(action, 0.0)
            target = reward + self.config.gamma * next_max
            state_q[action] = current_q + self.config.alpha * (target - current_q)

            next_max = max(state_q.values()) if state_q else 0.0
            reward = 0.0


class FormulaScorer:
    def __init__(
        self,
        panel: pd.DataFrame,
        split_config: SplitConfig,
        top_n: int,
        config: RLSearchConfig,
    ) -> None:
        self.panel = self._prepare_panel(panel)
        self.split_config = split_config
        self.top_n = top_n
        self.config = config
        self.formula_cache: dict[str, dict[str, Any]] = {}
        self.pool_cache: dict[tuple[str, ...], dict[str, Any]] = {}

        trade_date = self.panel["trade_date"]
        self.train = self.panel.loc[trade_date <= split_config.train_end].copy()
        self.val = self.panel.loc[(trade_date > split_config.train_end) & (trade_date <= split_config.val_end)].copy()
        self.test = self.panel.loc[(trade_date > split_config.val_end) & (trade_date <= split_config.test_end)].copy()
        self.latest = self.panel.loc[trade_date == int(trade_date.max())].copy()
        self.split_frames = {
            "train": self.train,
            "val": self.val,
            "test": self.test,
            "latest": self.latest,
        }
        self.factor_value_cache: dict[tuple[str, str], pd.Series] = {}
        self.target_cache = {
            name: frame[TARGET_COLUMN].groupby(frame["trade_date"]).transform(self._normalize_cross_section).fillna(0.0)
            for name, frame in self.split_frames.items()
            if name != "latest"
        }

    def score_formula(self, formula: str) -> dict[str, Any]:
        if formula in self.formula_cache:
            return self.formula_cache[formula]

        train_score = self._evaluate_formula(self.train, formula)
        val_score = self._evaluate_formula(self.val, formula)
        test_score = self._evaluate_formula(self.test, formula)

        self.formula_cache[formula] = {
            "formula": formula,
            "train_rank_ic": train_score,
            "val_rank_ic": val_score,
            "test_rank_ic": test_score,
            "selection_score": self._selection_score(train_score=train_score, val_score=val_score),
        }
        return self.formula_cache[formula]

    def build_summary(self, best_formula: str | None, pool_formulas: list[str] | None = None) -> dict[str, Any]:
        formulas = self._canonical_pool(pool_formulas or ([best_formula] if best_formula else []))
        summary = dict(self._optimize_pool(formulas))
        summary["baselines"] = self.evaluate_baselines()
        return summary

    def formula_score_table(self) -> pd.DataFrame:
        rows = [dict(values) for values in self.formula_cache.values()]
        rows = sorted(rows, key=lambda item: item["selection_score"], reverse=True)
        return pd.DataFrame(rows)

    def rank_latest_stocks(self, formula: str, reference_score: float) -> list[dict[str, Any]]:
        latest_scores = self._normalized_factor_for_split("latest", formula)
        return self._rank_latest_scores(latest_scores, reference_score=reference_score)

    def _rank_latest_scores(self, factor_scores: pd.Series, reference_score: float) -> list[dict[str, Any]]:
        latest = self.latest[["ts_code", "trade_date", "close_adj"]].copy()
        latest["factor_value"] = factor_scores
        latest = latest.replace([np.inf, -np.inf], np.nan).dropna(subset=["factor_value"])
        ascending = reference_score < 0
        latest = latest.sort_values(["factor_value", "ts_code"], ascending=[ascending, True]).reset_index(drop=True)
        latest["rank"] = latest.index + 1

        return [
            {
                "rank": int(row["rank"]),
                "ts_code": str(row["ts_code"]),
                "trade_date": int(row["trade_date"]),
                "close_adj": float(row["close_adj"]),
                "factor_value": float(row["factor_value"]),
            }
            for _, row in latest.head(self.top_n).iterrows()
        ]

    def evaluate_baselines(self) -> list[dict[str, Any]]:
        baseline_formulas = [
            "return_1d",
            "return_5d",
            "intraday_strength",
            "trend_gap",
        ]
        results: list[dict[str, Any]] = []
        for formula in baseline_formulas:
            results.append(dict(self.score_formula(formula)))
        return sorted(results, key=lambda item: item["val_rank_ic"], reverse=True)

    def _evaluate_formula(self, frame: pd.DataFrame, formula: str) -> float:
        split_name = self._frame_name(frame)
        factor = self._normalized_factor_for_split(split_name, formula)
        evaluation = frame[["trade_date", TARGET_COLUMN]].copy()
        evaluation["prediction"] = factor
        value = mean_daily_rank_ic(evaluation, prediction_col="prediction", target_col=TARGET_COLUMN)
        if pd.isna(value):
            return -1.0
        return float(value)

    @staticmethod
    def _compute_factor(frame: pd.DataFrame, formula: str) -> pd.Series:
        local_dict = {column: frame[column] for column in RL_FEATURE_COLUMNS}
        factor = pd.eval(formula, local_dict=local_dict, engine="python")
        return pd.Series(factor, index=frame.index, dtype=float).replace([np.inf, -np.inf], np.nan)

    def _normalized_factor(self, frame: pd.DataFrame, formula: str) -> pd.Series:
        split_name = self._frame_name(frame)
        return self._normalized_factor_for_split(split_name, formula)

    def _normalized_factor_for_split(self, split_name: str, formula: str) -> pd.Series:
        key = (split_name, formula)
        if key in self.factor_value_cache:
            return self.factor_value_cache[key].copy()
        frame = self.split_frames[split_name]
        raw = self._compute_factor(frame, formula)
        normalized = raw.groupby(frame["trade_date"]).transform(self._normalize_cross_section)
        series = pd.Series(normalized, index=frame.index, dtype=float)
        self.factor_value_cache[key] = series
        return series.copy()

    @staticmethod
    def _normalize_cross_section(values: pd.Series) -> pd.Series:
        values = values.replace([np.inf, -np.inf], np.nan)
        if values.notna().sum() < 3:
            return pd.Series(np.nan, index=values.index, dtype=float)

        lower = values.quantile(0.01)
        upper = values.quantile(0.99)
        clipped = values.clip(lower, upper)
        std = clipped.std()
        if pd.isna(std) or std == 0:
            return pd.Series(np.nan, index=values.index, dtype=float)
        return pd.Series((clipped - clipped.mean()) / std, index=values.index, dtype=float)

    def evaluate_formula_in_pool(self, formula: str, current_pool: list[str]) -> dict[str, Any]:
        current_summary = self._optimize_pool(self._canonical_pool(current_pool))
        candidate_pool = self._canonical_pool([*current_pool, formula])
        updated_summary = self._optimize_pool(candidate_pool)
        reward = updated_summary["selection_score"] - current_summary["selection_score"]
        return {
            "reward": float(reward),
            "current_summary": current_summary,
            "updated_summary": updated_summary,
            "updated_pool": [item["formula"] for item in updated_summary["factor_pool"]],
        }

    def _optimize_pool(self, formulas: list[str]) -> dict[str, Any]:
        canonical = tuple(self._canonical_pool(formulas))
        if canonical in self.pool_cache:
            return dict(self.pool_cache[canonical])

        if not canonical:
            empty_summary = {
                "formula": "N/A",
                "best_formula": "N/A",
                "train_rank_ic": 0.0,
                "val_rank_ic": 0.0,
                "test_rank_ic": 0.0,
                "selection_score": 0.0,
                "top_stocks": [],
                "factor_pool": [],
                "representative_formula_stats": {},
            }
            self.pool_cache[canonical] = dict(empty_summary)
            return empty_summary

        selected_formulas = self._select_diverse_formulas(list(canonical))
        train_matrix = self._formula_matrix(self.train, selected_formulas)
        target = self._normalized_target(self.train)
        weights = self._fit_pool_weights(train_matrix, target)

        pruned_formulas, pruned_weights = self._prune_pool(selected_formulas, weights)
        train_score = self._evaluate_pool(self.train, pruned_formulas, pruned_weights)
        val_score = self._evaluate_pool(self.val, pruned_formulas, pruned_weights)
        test_score = self._evaluate_pool(self.test, pruned_formulas, pruned_weights)
        latest_scores = self._pool_factor(self.latest, pruned_formulas, pruned_weights)
        top_stocks = self._rank_latest_scores(latest_scores, reference_score=val_score)

        pool_entries: list[dict[str, Any]] = []
        for formula, weight in zip(pruned_formulas, pruned_weights):
            formula_stats = self.score_formula(formula)
            pool_entries.append(
                {
                    "formula": formula,
                    "weight": float(weight),
                    "train_rank_ic": float(formula_stats["train_rank_ic"]),
                    "val_rank_ic": float(formula_stats["val_rank_ic"]),
                    "test_rank_ic": float(formula_stats["test_rank_ic"]),
                    "selection_score": float(formula_stats["selection_score"]),
                }
            )

        representative_formula = max(pool_entries, key=lambda item: abs(item["weight"]))["formula"] if pool_entries else str(canonical[0])
        summary = {
            "formula": representative_formula,
            "best_formula": representative_formula,
            "train_rank_ic": float(train_score),
            "val_rank_ic": float(val_score),
            "test_rank_ic": float(test_score),
            "selection_score": float(self._selection_score(train_score=train_score, val_score=val_score)),
            "top_stocks": top_stocks,
            "factor_pool": pool_entries,
            "representative_formula_stats": dict(self.score_formula(representative_formula)),
        }
        self.pool_cache[canonical] = dict(summary)
        return summary

    def _select_diverse_formulas(self, formulas: list[str]) -> list[str]:
        selected: list[str] = []
        selected_series: list[pd.Series] = []
        ordered = sorted(formulas, key=lambda item: self.score_formula(item)["selection_score"], reverse=True)
        for formula in ordered:
            candidate = self._normalized_factor_for_split("val", formula)
            if candidate.notna().sum() < 10:
                continue
            too_similar = False
            for existing in selected_series:
                corr = candidate.corr(existing, method="spearman")
                if pd.notna(corr) and abs(float(corr)) >= self.config.max_formula_correlation:
                    too_similar = True
                    break
            if too_similar:
                continue
            selected.append(formula)
            selected_series.append(candidate)
        return selected or ordered[:1]

    def _formula_matrix(self, frame: pd.DataFrame, formulas: list[str]) -> np.ndarray:
        split_name = self._frame_name(frame)
        columns = [
            self._normalized_factor_for_split(split_name, formula).fillna(0.0).to_numpy(dtype=np.float64)
            for formula in formulas
        ]
        if not columns:
            return np.zeros((len(frame), 0), dtype=np.float64)
        return np.column_stack(columns)

    def _normalized_target(self, frame: pd.DataFrame) -> np.ndarray:
        split_name = self._frame_name(frame)
        return self.target_cache[split_name].to_numpy(dtype=np.float64)

    @staticmethod
    def _fit_pool_weights(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
        if matrix.size == 0:
            return np.zeros(0, dtype=np.float64)
        ridge = 1e-3
        lhs = matrix.T @ matrix + ridge * np.eye(matrix.shape[1], dtype=np.float64)
        rhs = matrix.T @ target
        return np.linalg.solve(lhs, rhs)

    def _prune_pool(self, formulas: list[str], weights: np.ndarray) -> tuple[list[str], np.ndarray]:
        if len(formulas) <= self.config.pool_size:
            return formulas, weights
        order = np.argsort(np.abs(weights))[::-1][: self.config.pool_size]
        kept = sorted(int(index) for index in order)
        return [formulas[index] for index in kept], weights[kept]

    def _evaluate_pool(self, frame: pd.DataFrame, formulas: list[str], weights: np.ndarray) -> float:
        factor = self._pool_factor(frame, formulas, weights)
        evaluation = frame[["trade_date", TARGET_COLUMN]].copy()
        evaluation["prediction"] = factor
        value = mean_daily_rank_ic(evaluation, prediction_col="prediction", target_col=TARGET_COLUMN)
        if pd.isna(value):
            return -1.0
        return float(value)

    def _pool_factor(self, frame: pd.DataFrame, formulas: list[str], weights: np.ndarray) -> pd.Series:
        if not formulas:
            return pd.Series(np.nan, index=frame.index, dtype=float)
        matrix = self._formula_matrix(frame, formulas)
        values = matrix @ weights
        return pd.Series(values, index=frame.index, dtype=float)

    @staticmethod
    def _canonical_pool(formulas: list[str]) -> list[str]:
        return sorted({formula for formula in formulas if formula})

    def _frame_name(self, frame: pd.DataFrame) -> str:
        for name, candidate in self.split_frames.items():
            if frame is candidate:
                return name
        raise ValueError("Unknown frame reference passed to FormulaScorer.")

    @staticmethod
    def _prepare_panel(panel: pd.DataFrame) -> pd.DataFrame:
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
        working = working.replace([np.inf, -np.inf], np.nan)
        return working

    def _selection_score(self, train_score: float, val_score: float) -> float:
        return (
            self.config.train_reward_weight * train_score
            + self.config.validation_reward_weight * val_score
            - self.config.stability_penalty_weight * abs(train_score - val_score)
        )


def run_rl_pipeline(
    panel: pd.DataFrame,
    output_dir: Path,
    split_config: SplitConfig,
    config: RLSearchConfig,
) -> dict[str, Any]:
    ensure_directory(output_dir)

    scorer = FormulaScorer(panel=panel, split_config=split_config, top_n=config.top_n, config=config)
    agent = SymbolicFactorRLAgent(config=config)
    history, summary, all_scores = agent.train(scorer=scorer)

    history.to_csv(output_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    all_scores.to_csv(output_dir / "formula_scores.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary["top_stocks"]).to_csv(output_dir / "top_stocks.csv", index=False, encoding="utf-8-sig")
    save_json(summary, output_dir / "summary.json")
    _plot_training_curve(history, output_dir / "training_curve.png")

    return {"history": history, "summary": summary, "all_scores": all_scores}


def _plot_training_curve(history: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(history["episode"], history["selection_score"], label="episode selection score", alpha=0.5)
    plt.plot(history["episode"], history["selected_best_score"], label="best selection score")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("RL factor search progression")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

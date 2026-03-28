from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, url_for

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from integrated_alpha.common.config import ProjectPaths
from integrated_alpha.common.io_utils import load_json
from integrated_alpha.dashboard_module.service import (
    explain_stock_pick,
    get_dashboard_bundle,
    get_price_demo_bundle,
    refresh_dashboard_bundle,
)
from integrated_alpha.llm_module.claude_chat import ClaudeChatSession

app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "templates"),
    static_folder=str(PROJECT_ROOT / "static"),
)


@app.route("/")
def home() -> str:
    bundle = get_dashboard_bundle(PROJECT_ROOT)
    recommendations = [
        {
            "recommendation_rank": int(row["recommendation_rank"]),
            "ts_code": str(row["ts_code"]),
            "predicted_future_return_20d": float(row["predicted_future_return_20d"]),
            "factor_value": float(row["factor_value"]),
            "combined_score": float(row["combined_score"]),
        }
        for _, row in bundle["top_recommendations"].iterrows()
    ]
    return render_template(
        "dashboard.html",
        summary=bundle["summary"],
        recommendations=recommendations,
    )


@app.route("/stock/<stock_code>")
def stock_detail(stock_code: str) -> str:
    bundle = get_dashboard_bundle(PROJECT_ROOT)
    detail = explain_stock_pick(bundle, stock_code)
    price_demo = get_price_demo_bundle(PROJECT_ROOT, stock_code)
    price_fit_chart = _line_chart_base64(
        frame=price_demo["prediction_frame"],
        x_col="trade_date",
        series=[
            ("actual_price", "Actual Price"),
            ("predicted_price", "Predicted Price"),
        ],
        title=f"{stock_code} Single-Stock Price Fit",
        y_label="Price",
    )
    alpha_fit_chart = None
    if not detail["fit_frame"].empty:
        alpha_fit_chart = _line_chart_base64(
            frame=detail["fit_frame"],
            x_col="trade_date",
            series=[
                ("target", "Actual future_return_20d"),
                ("prediction", "Predicted future_return_20d"),
            ],
            title=f"{stock_code} Historical LSTM Fit",
            y_label="Return",
        )

    return render_template(
        "stock_detail.html",
        summary=bundle["summary"],
        detail=detail,
        price_fit_chart=price_fit_chart,
        alpha_fit_chart=alpha_fit_chart,
        price_demo_summary=price_demo["summary"],
    )


@app.route("/refresh", methods=["POST"])
def refresh() -> Any:
    refresh_dashboard_bundle(PROJECT_ROOT)
    return redirect(url_for("home"))


@app.route("/api/chat", methods=["POST"])
def api_chat() -> Any:
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question", "")).strip()
    stock_code = str(payload.get("stock_code", "")).strip()
    if not question:
        return jsonify({"ok": False, "error": "Question is empty."}), 400

    bundle = get_dashboard_bundle(PROJECT_ROOT)
    context = {
        "top_recommendations": [
            {
                "rank": int(row["recommendation_rank"]),
                "ts_code": str(row["ts_code"]),
                "predicted_future_return_20d": float(row["predicted_future_return_20d"]),
                "combined_score": float(row["combined_score"]),
            }
            for _, row in bundle["top_recommendations"].iterrows()
        ]
    }
    if stock_code:
        detail = explain_stock_pick(bundle, stock_code)
        context["selected_stock"] = {
            "ts_code": detail["stock_code"],
            "recommendation_rank": int(detail["recommendation_rank"]),
            "predicted_future_return_20d": float(detail["predicted_future_return_20d"]),
            "factor_value": float(detail["factor_value"]),
            "formula": detail["formula"],
            "reasons": detail["reasons"],
        }

    try:
        paths = ProjectPaths.from_root(PROJECT_ROOT)
        chat_session = ClaudeChatSession.from_env(output_dir=paths.llm_dir)
        answer = chat_session.ask(
            question=f"{question}\n\nDashboard context:\n{json.dumps(context, ensure_ascii=False, indent=2)}",
            experiment_summary=_chat_summary(paths),
        )
    except RuntimeError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    return jsonify({"ok": True, "answer": answer})


def _chat_summary(paths: ProjectPaths) -> dict[str, Any]:
    summary_path = paths.combined_dir / "experiment_summary.json"
    if summary_path.exists():
        return load_json(summary_path)

    bundle = get_dashboard_bundle(PROJECT_ROOT)
    return {
        "data_summary": {
            "stock_count": int(bundle["summary"]["market_stock_count"]),
            "row_count": int(len(bundle["panel"])),
        },
        "rl_summary": {
            "best_formula": str(bundle["summary"]["rl_formula"]),
            "val_rank_ic": float(bundle["summary"]["rl_validation_rank_ic"]),
            "test_rank_ic": float(bundle["summary"]["rl_test_rank_ic"]),
        },
        "lstm_summary": {
            "rmse": float(bundle["summary"]["lstm_rmse"]),
            "mean_daily_rank_ic": float(bundle["summary"]["lstm_mean_daily_rank_ic"]),
            "baseline_mean_daily_rank_ic": 0.0,
        },
    }


def _line_chart_base64(
    frame: pd.DataFrame,
    x_col: str,
    series: list[tuple[str, str]],
    title: str,
    y_label: str,
) -> str:
    chart_frame = frame.copy()
    chart_frame[x_col] = pd.to_datetime(chart_frame[x_col].astype(str), format="%Y%m%d")

    fig, ax = plt.subplots(figsize=(10, 4))
    for column, label in series:
        ax.plot(chart_frame[x_col], chart_frame[column], label=label, linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("Trade date")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140)
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Alpha Stock dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()

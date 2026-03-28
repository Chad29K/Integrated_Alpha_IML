from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import os

from anthropic import Anthropic
from anthropic import APIError
from anthropic import BadRequestError
from dotenv import load_dotenv

from integrated_alpha.common.io_utils import ensure_directory, save_text


@dataclass
class ClaudeChatSession:
    client: Anthropic
    model: str
    output_dir: Path
    history: list[dict[str, str]]

    @classmethod
    def from_env(cls, output_dir: Path) -> "ClaudeChatSession":
        env_path = output_dir.parent.parent / ".env"
        load_dotenv(env_path)

        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest").strip()

        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is missing. Copy .env.example to .env and add the API key."
            )

        ensure_directory(output_dir / "transcripts")
        return cls(client=Anthropic(api_key=api_key), model=model, output_dir=output_dir, history=[])

    def ask(self, question: str, experiment_summary: dict[str, Any]) -> str:
        system_prompt = self._build_system_prompt(experiment_summary)
        messages = self.history + [{"role": "user", "content": question}]
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=900,
                system=system_prompt,
                messages=messages,
            )
        except BadRequestError as exc:
            raise RuntimeError(f"Chat request failed: {exc}") from exc
        except APIError as exc:
            raise RuntimeError(f"Chat API error: {exc}") from exc
        answer = "".join(block.text for block in response.content if hasattr(block, "text")).strip()
        self.history.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        )
        self._save_transcript()
        return answer

    def interactive_loop(self, experiment_summary: dict[str, Any]) -> None:
        print("Chat assistant is ready. Type 'exit' to quit.")
        while True:
            question = input("You> ").strip()
            if not question:
                continue
            if question.lower() in {"exit", "quit"}:
                break
            answer = self.ask(question=question, experiment_summary=experiment_summary)
            print(f"\nAssistant> {answer}\n")

    def _build_system_prompt(self, experiment_summary: dict[str, Any]) -> str:
        data = experiment_summary["data_summary"]
        rl = experiment_summary["rl_summary"]
        lstm = experiment_summary["lstm_summary"]

        return "\n".join(
            [
                "You are the LLM module of Alpha Stock, an A-share stock selection system.",
                "Answer in clear Chinese by default unless the user asks for English.",
                "Only use the system information provided below.",
                "Be honest about limitations and avoid fabricating data.",
                "",
                "System summary:",
                f"- Data stocks: {data['stock_count']}",
                f"- Data rows: {data['row_count']}",
                f"- RL best formula: {rl['best_formula']}",
                f"- RL validation Rank IC: {rl['val_rank_ic']:.6f}",
                f"- RL test Rank IC: {rl['test_rank_ic']:.6f}",
                f"- LSTM RMSE: {lstm['rmse']:.6f}",
                f"- LSTM daily Rank IC: {lstm['mean_daily_rank_ic']:.6f}",
                f"- LSTM baseline daily Rank IC: {lstm['baseline_mean_daily_rank_ic']:.6f}",
            ]
        )

    def _save_transcript(self) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        transcript_path = self.output_dir / "transcripts" / f"chat_{timestamp}.md"
        lines = ["# Chat Transcript", ""]
        for turn in self.history:
            prefix = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"## {prefix}")
            lines.append(turn["content"])
            lines.append("")
        save_text("\n".join(lines), transcript_path)

"""Abstract base class for all LLM wrappers."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class LLMResponse:
    model_name: str
    question: str
    raw_response: str
    reasoning_steps: list[str]
    final_answer: str
    run_index: int = 0


class BaseLLM(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def _call_api(self, system_prompt: str, user_message: str) -> str: ...

    def query(self, question: str, system_prompt: str, run_index: int = 0) -> LLMResponse:
        raw = self._call_api(system_prompt, question)
        steps, answer = self._parse_response(raw)
        return LLMResponse(
            model_name=self.model_name,
            question=question,
            raw_response=raw,
            reasoning_steps=steps,
            final_answer=answer,
            run_index=run_index,
        )

    @staticmethod
    def _parse_response(text: str) -> tuple[list[str], str]:
        lines = text.strip().splitlines()
        steps: list[str] = []
        final_answer = ""
        in_reasoning = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            lower = stripped.lower()

            if lower.startswith("final answer:") or lower.startswith("answer:"):
                final_answer = stripped.split(":", 1)[-1].strip()
                in_reasoning = False
                continue

            if lower.startswith("reasoning:"):
                in_reasoning = True
                continue

            if lower.startswith("step ") and ":" in stripped:
                steps.append(stripped.split(":", 1)[-1].strip())
                in_reasoning = True
                continue

            if re.match(r"^\d+[.)]\s", stripped):
                steps.append(re.sub(r"^\d+[.)]\s+", "", stripped))
                in_reasoning = True
                continue

            if in_reasoning:
                steps.append(stripped)

        if not steps:
            sentences = re.split(r"(?<=[.!?])\s+", text.strip())
            steps = [s for s in sentences if len(s) > 20][:6]

        if not final_answer and lines:
            final_answer = lines[-1].strip()

        return steps, final_answer

"""Orchestrates querying LLMs and computing all five metrics per question."""

from dataclasses import dataclass, asdict

from tqdm import tqdm

from config import COT_SYSTEM_PROMPT, DEFAULT_WEIGHTS, T_RUNS
from llm.base import BaseLLM, LLMResponse
from evaluation.metrics import (
    compute_aas, compute_ras, compute_slms,
    compute_cs, compute_dkus, compute_fps,
)


@dataclass
class QuestionResult:
    question_id: str
    model_name: str
    aas: float
    ras: float
    slms: float
    cs: float
    dkus: float
    fps: float
    runs: list  # serialised LLMResponse dicts


@dataclass
class EvaluationResult:
    model_name: str
    question_results: list[QuestionResult]
    mean_aas: float = 0.0
    mean_ras: float = 0.0
    mean_slms: float = 0.0
    mean_cs: float = 0.0
    mean_dkus: float = 0.0
    mean_fps: float = 0.0


def evaluate_model(
    llm: BaseLLM,
    golden_truth: list[dict],
    weights: dict = None,
    t_runs: int = None,
) -> EvaluationResult:
    weights = weights or DEFAULT_WEIGHTS
    t_runs = t_runs or T_RUNS
    question_results: list[QuestionResult] = []

    for gt in tqdm(golden_truth, desc=f"Evaluating {llm.model_name}"):
        question = gt["question"]
        golden_steps = [s["description"] for s in gt["golden_reasoning_steps"]]
        golden_reasoning = " ".join(golden_steps)
        golden_answer = gt["golden_final_answer"]
        must_include = gt["evaluation_criteria"]["must_include_concepts"]

        runs: list[LLMResponse] = [
            llm.query(question, COT_SYSTEM_PROMPT, run_index=t)
            for t in range(t_runs)
        ]

        first = runs[0]
        model_reasoning = " ".join(first.reasoning_steps)

        aas = compute_aas(first.final_answer, golden_answer)
        ras = compute_ras(model_reasoning, golden_reasoning)
        slms = compute_slms(first.reasoning_steps, golden_steps)
        ras_scores = [
            compute_ras(" ".join(r.reasoning_steps), golden_reasoning)
            for r in runs
        ]
        cs = compute_cs(ras_scores)
        dkus = compute_dkus(model_reasoning, must_include)
        fps = compute_fps(aas, ras, slms, cs, dkus, weights)

        question_results.append(QuestionResult(
            question_id=gt["question_id"],
            model_name=llm.model_name,
            aas=aas,
            ras=ras,
            slms=slms,
            cs=cs,
            dkus=dkus,
            fps=fps,
            runs=[{
                "run_index": r.run_index,
                "reasoning_steps": r.reasoning_steps,
                "final_answer": r.final_answer,
                "raw_response": r.raw_response,
            } for r in runs],
        ))

    n = len(question_results)
    return EvaluationResult(
        model_name=llm.model_name,
        question_results=question_results,
        mean_aas=sum(r.aas for r in question_results) / n,
        mean_ras=sum(r.ras for r in question_results) / n,
        mean_slms=sum(r.slms for r in question_results) / n,
        mean_cs=sum(r.cs for r in question_results) / n,
        mean_dkus=sum(r.dkus for r in question_results) / n,
        mean_fps=sum(r.fps for r in question_results) / n,
    )


def results_to_dict(result: EvaluationResult) -> dict:
    return {
        "model_name": result.model_name,
        "mean_aas": result.mean_aas,
        "mean_ras": result.mean_ras,
        "mean_slms": result.mean_slms,
        "mean_cs": result.mean_cs,
        "mean_dkus": result.mean_dkus,
        "mean_fps": result.mean_fps,
        "question_results": [asdict(qr) for qr in result.question_results],
    }

"""Orchestrates querying LLMs and computing all five metrics per question.

Persistent caching: LLM responses (and their reasoning embeddings) are cached
per (question_id, model_name, run_index) in ChromaDB. Re-running the same model
on the same questions skips the API calls entirely. The cache is keyed by a
hash of the question text, so it auto-invalidates if the dataset is regenerated.
"""

from dataclasses import dataclass, asdict, field

from tqdm import tqdm

from config import COT_SYSTEM_PROMPT, DEFAULT_WEIGHTS, T_RUNS
from llm.base import BaseLLM, LLMResponse
from embeddings.encoder import encode
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
    cache_hits: int = 0
    cache_misses: int = 0


def evaluate_model(
    llm: BaseLLM,
    golden_truth: list[dict],
    weights: dict = None,
    t_runs: int = None,
    use_cache: bool = True,
) -> EvaluationResult:
    weights = weights or DEFAULT_WEIGHTS
    t_runs = t_runs or T_RUNS
    question_results: list[QuestionResult] = []
    cache_hits = 0
    cache_misses = 0

    store = None
    if use_cache:
        try:
            from storage.chroma_store import ChromaStore
            store = ChromaStore()
        except Exception as e:
            tqdm.write(f"[cache] disabled: {e}")
            store = None

    for gt in tqdm(golden_truth, desc=f"Evaluating {llm.model_name}"):
        question = gt["question"]
        qid = gt["question_id"]
        golden_steps = [s["description"] for s in gt["golden_reasoning_steps"]]
        golden_reasoning = " ".join(golden_steps)
        golden_answer = gt["golden_final_answer"]
        must_include = gt["evaluation_criteria"]["must_include_concepts"]

        runs: list[LLMResponse] = []
        run_reasoning_embs: list = []  # one per run, used for RAS

        for t in range(t_runs):
            cached = store.get_response(qid, question, llm.model_name, t) if store else None
            if cached is not None:
                cache_hits += 1
                runs.append(LLMResponse(
                    model_name=llm.model_name,
                    question=question,
                    raw_response=cached["raw_response"],
                    reasoning_steps=cached["reasoning_steps"],
                    final_answer=cached["final_answer"],
                    run_index=t,
                ))
                run_reasoning_embs.append(cached["reasoning_embedding"])
            else:
                cache_misses += 1
                r = llm.query(question, COT_SYSTEM_PROMPT, run_index=t)
                reasoning_text = " ".join(r.reasoning_steps)
                emb = encode(reasoning_text) if reasoning_text else None
                if store and emb is not None:
                    store.store_response(
                        question_id=qid,
                        question_text=question,
                        model_name=llm.model_name,
                        run_index=t,
                        reasoning_steps=r.reasoning_steps,
                        final_answer=r.final_answer,
                        raw_response=r.raw_response,
                        reasoning_embedding=emb.tolist(),
                    )
                runs.append(r)
                run_reasoning_embs.append(emb)

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
            question_id=qid,
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
        cache_hits=cache_hits,
        cache_misses=cache_misses,
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
        "cache_hits": result.cache_hits,
        "cache_misses": result.cache_misses,
        "question_results": [asdict(qr) for qr in result.question_results],
    }

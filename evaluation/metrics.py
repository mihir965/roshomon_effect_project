"""Pure metric functions for the evaluation framework.

All scores are in [0, 1].
"""

import numpy as np
from embeddings.encoder import encode, cosine_similarity


def compute_aas(model_answer: str, golden_answer: str) -> float:
    """Answer Accuracy Score: embedding cosine similarity between final answers."""
    if not model_answer or not golden_answer:
        return 0.0
    return max(0.0, cosine_similarity(encode(model_answer), encode(golden_answer)))


def compute_ras(model_reasoning: str, golden_reasoning: str) -> float:
    """Reasoning Alignment Score: cosine similarity of full reasoning embeddings."""
    if not model_reasoning or not golden_reasoning:
        return 0.0
    return max(0.0, cosine_similarity(encode(model_reasoning), encode(golden_reasoning)))


def compute_slms(model_steps: list[str], golden_steps: list[str]) -> float:
    """Step-wise Logical Matching Score.

    For each golden step, find the maximum cosine similarity across all model
    steps, then average over golden steps.  Penalises missing coverage.
    """
    if not golden_steps or not model_steps:
        return 0.0

    golden_embs = encode(golden_steps)   # shape [n, dim]
    model_embs = encode(model_steps)     # shape [m, dim]

    # golden_embs @ model_embs.T  -> shape [n, m]
    sim_matrix = golden_embs @ model_embs.T
    return float(sim_matrix.max(axis=1).mean())


def compute_cs(ras_scores: list[float]) -> float:
    """Consistency Score: 1 - variance of RAS across T runs.

    Low variance => stable reasoning => high CS.
    """
    if len(ras_scores) < 2:
        return 1.0
    return max(0.0, 1.0 - float(np.var(ras_scores)))


def compute_dkus(model_reasoning: str, must_include_concepts: list[str]) -> float:
    """Domain Knowledge Utilization Score: fraction of required concepts mentioned."""
    if not must_include_concepts:
        return 1.0
    text = model_reasoning.lower()
    covered = sum(1 for c in must_include_concepts if c.lower() in text)
    return covered / len(must_include_concepts)


def compute_fps(aas: float, ras: float, slms: float, cs: float, dkus: float,
                weights: dict) -> float:
    """Final Composite Performance Score (weighted sum, weights must sum to 1)."""
    return (
        weights["w_aas"] * aas
        + weights["w_ras"] * ras
        + weights["w_slms"] * slms
        + weights["w_cs"] * cs
        + weights["w_dkus"] * dkus
    )

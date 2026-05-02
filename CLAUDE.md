# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Evaluation and Recommendation Framework for Large Language Models Based on Reasoning and Answer Alignment** (CS671, Team MVP — Varun, Pranika, Mihir)

This framework evaluates multiple LLMs not just on final answer accuracy but on *reasoning quality*, exposing the Rashomon Effect: models that achieve the same answer via different (possibly flawed) reasoning chains. We measure **reasoning surface alignment** against a golden truth dataset using five custom metrics.

## Architecture

### Pipeline Flow
```
HuggingFace Dataset
    ↓ data/dataset_loader.py
Golden Truth JSON (data/golden_truth.json)
    ↓ evaluation/evaluator.py
Multiple LLMs queried with CoT prompting (llm/)
    ↓
Embeddings computed (embeddings/encoder.py)
    ↓
Five metrics computed (evaluation/metrics.py)
    ↓
Results persisted (storage/chroma_store.py)
    ↓
Streamlit dashboard (app.py)
```

### Scoring Metrics

| Metric | Formula | What it measures |
|--------|---------|-----------------|
| **AAS** | cosine(E(model_answer), E(golden_answer)) | Final answer correctness |
| **RAS** | cosine(E(Rᵢ), E(Rg)) | Full reasoning chain alignment |
| **SLMS** | Σₖ maxⱼ Sim(sₖ, rⱼ) / n | Step-level logical coverage of golden steps |
| **CS** | 1 − Var(RAS¹…RASᵀ) | Consistency across T repeated runs |
| **DKUS** | \|Cg ∩ Cᵢ\| / \|Cg\| | Domain concept coverage |
| **FPS** | w₁·AAS + w₂·RAS + w₃·SLMS + w₄·CS + w₅·DKUS | Weighted composite (weights configurable) |

### Golden Truth JSON Schema
```json
{
  "question_id": "q_0001",
  "task_domain": "hard-reasoning",
  "difficulty_level": "hard",
  "question": "...",
  "golden_reasoning_steps": [{"step_number": 1, "description": "..."}],
  "golden_final_answer": "...",
  "expected_knowledge_domain": "...",
  "reasoning_type": "multi-step",
  "evaluation_criteria": {
    "must_include_concepts": [],
    "common_failure_modes": [],
    "logical_dependencies": []
  }
}
```

### Dataset
HuggingFace: `avemio/German-RAG-ORPO-Alpaca-HESSIAN-AI`, split `hard-reasoning-en`

## Tech Stack
- **Frontend**: Streamlit (`app.py`) with Plotly charts
- **LLM APIs**: OpenAI, Anthropic, Google Gemini — all implement `llm/base.py:BaseLLM`
- **Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`) via `embeddings/encoder.py`
- **Storage**: ChromaDB (`storage/chroma_store.py`) — persists embeddings and scores to avoid recomputation
- **Data**: HuggingFace `datasets`, Pandas

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app (main entry point)
streamlit run app.py

# Run evaluation pipeline via CLI
python pipeline.py --models openai anthropic gemini --n_questions 20 --t_runs 3

# Build golden truth dataset only
python pipeline.py --rebuild_golden --n_questions 50 --models openai

# Inspect the HuggingFace dataset
python data/dataset_loader.py
```

## Configuration

Copy `.env.example` to `.env` and populate:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`

Model names, default FPS weights, T_RUNS, and embedding model are all in `config.py`.

## Key Design Decisions
- RAS uses cosine similarity on **full-text embeddings** of the concatenated reasoning chain, not per-step
- SLMS is asymmetric: for each **golden** step, find the best-matching **model** step (not vice versa) — penalizes missing coverage
- CS requires T repeated runs of the same query; default T=3 to balance API cost vs. reliability signal
- DKUS uses simple substring matching of `must_include_concepts` against model reasoning (lowercased)
- FPS weights are user-adjustable in the Streamlit sidebar in real-time without re-running evaluation
- ChromaDB persistence means embeddings are computed once and reused across sessions
- All LLMs use the same `COT_SYSTEM_PROMPT` (defined in `config.py`) for fair comparison

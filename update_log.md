# Update Log

A running log of changes made to this project. Add a new entry at the top each time work is done.

---

## [2026-05-02] — Initial project scaffold

**By:** Claude Code

**Summary:** Built the full project skeleton from scratch based on the project proposal (`Project_update_ppt.pdf`). No code existed before this session.

**Files created:**
- `CLAUDE.md` — architecture guide and command reference for future Claude sessions
- `requirements.txt` — all Python dependencies
- `.env.example` — template for API keys
- `config.py` — central config (API keys, model names, default FPS weights, CoT system prompt)
- `data/dataset_loader.py` — loads `avemio/German-RAG-ORPO-Alpaca-HESSIAN-AI` (`hard-reasoning-en` split) from HuggingFace and converts rows into the golden truth JSON schema
- `llm/base.py` — abstract `BaseLLM` class with shared CoT response parser
- `llm/openai_llm.py` — OpenAI (GPT-4o) wrapper
- `llm/anthropic_llm.py` — Anthropic (Claude 3.5 Sonnet) wrapper
- `llm/gemini_llm.py` — Google Gemini 1.5 Pro wrapper
- `embeddings/encoder.py` — singleton Sentence Transformers encoder (`all-MiniLM-L6-v2`) with cosine similarity helper
- `evaluation/metrics.py` — pure implementations of AAS, RAS, SLMS, CS, DKUS, FPS
- `evaluation/evaluator.py` — orchestrates T-run evaluation loop per model per question
- `storage/chroma_store.py` — ChromaDB persistence for embeddings and scores
- `pipeline.py` — CLI runner (`--models`, `--n_questions`, `--t_runs`, `--rebuild_golden`)
- `app.py` — Streamlit dashboard (Dataset / Evaluate / Results pages with live FPS weight sliders)

**Metrics implemented (from slides):**

| Metric | Formula |
|--------|---------|
| AAS | cosine(E(model\_answer), E(golden\_answer)) |
| RAS | cosine(E(model\_reasoning), E(golden\_reasoning)) |
| SLMS | Σₖ maxⱼ Sim(sₖ, rⱼ) / n |
| CS | 1 − Var(RAS¹ … RASᵀ) |
| DKUS | \|Cg ∩ Ci\| / \|Cg\| |
| FPS | w₁·AAS + w₂·RAS + w₃·SLMS + w₄·CS + w₅·DKUS |

**Known gaps / next steps:**
- Dataset column names not yet verified against the real HuggingFace dataset — `dataset_loader.py` auto-detects common names but may need manual adjustment
- ChromaDB storage not yet wired into the main evaluation loop (scores are saved to `results.json` but not persisted to Chroma automatically)
- No golden truth refinement step — concept extraction in `_extract_key_concepts()` uses simple regex, not an LLM
- No automated tests written yet

---

<!-- Template for future entries:

## [YYYY-MM-DD] — Short title

**By:** <name>

**Summary:** What was done and why.

**Files changed:**
- `path/to/file.py` — what changed

**Known gaps / next steps:**
- ...

-->

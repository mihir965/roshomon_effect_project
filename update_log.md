# Update Log

A running log of changes made to this project. Add a new entry at the top each time work is done.

---

## [2026-05-03] — Multi-model comparison, persistent cache, GUI overhaul

**By:** Mihir + Claude Code (branch: `mihir`)

**Goal:** make the framework production-usable for comparing many models (including multiple from the same provider), with a robust, GUI-driven workflow and incremental evaluation.

### Highlights

- **Compare any number of models from any provider in a single run.** Picker is populated *live* from each provider's API.
- **Persistent LLM response cache** in ChromaDB — re-running the same model on the same questions skips the API calls entirely, so iterating on weights/runs is free.
- **`results.json` accumulates across sessions** instead of getting overwritten — replaces by `model_name` so re-running one model doesn't wipe the others.
- **Migrated off the deprecated `google-generativeai` SDK** to the new `google-genai` SDK.
- **Three "fresh start" levels** in the GUI: clear leaderboard / wipe LLM cache / rebuild golden truth.

### Files changed

| File | What changed |
|---|---|
| `llm/gemini_llm.py` | Rewritten on top of `google-genai` (`genai.Client(...).models.generate_content`). Old `google.generativeai.GenerativeModel` API gone. |
| `llm/model_registry.py` | **NEW.** `list_openai_models()`, `list_anthropic_models()`, `list_gemini_models()`, `list_ollama_models()`, plus `parse_model_spec("provider:model_id")` and `model_exists(provider, model_id)`. Returns `[]` on missing key / network failure (never raises). |
| `pipeline.py` | Accepts `provider:model_id` form (e.g. `anthropic:claude-haiku-4-5-20251001`, `ollama:llama3:latest`). Pre-flight validates each spec against the live catalog. Merges new runs into `results.json` by `model_name` (use `--overwrite` to nuke). Stamps each entry with `provider`, `timestamp`, `n_questions`, `t_runs`. Prints cache hits/misses. |
| `app.py` | Per-provider 4-column multiselect picker, populated from registry with 2-min cache and a "🔄 Refresh model list" button. Selected models shown as `provider:model_id` chips. Save logic now reads-merge-writes `results.json` (replace by `model_name`). Results page top bar adds: **Drop selected**, **🗑 Clear all**, and **🧹 Cache (N)** for wiping cached LLM responses. Leaderboard now shows Provider, N, T, "Run at (UTC)" columns. Cache hits/misses shown in the per-model success banner. Replaced deprecated `use_container_width=True` with `width="stretch"`. |
| `evaluation/evaluator.py` | Cache lookup before each `llm.query` keyed by `(question_id, model_name, run_index)`. Cache hits skip the API entirely; misses store reasoning steps + final answer + raw response + reasoning embedding + question hash. Adds `cache_hits` / `cache_misses` to `EvaluationResult`. |
| `embeddings/encoder.py` | In-process LRU (4096 entries) around `encode()`. Within a session, golden text is embedded once instead of `n_runs × n_models` times. Batches uncached strings together. |
| `storage/chroma_store.py` | API rewritten to actually fit the cache use case: `store_response(question_id, question_text, model_name, run_index, reasoning_steps, final_answer, raw_response, reasoning_embedding)` and `get_response(...)` returning `None` on miss/text-changed. Added `count()` and `wipe()`. Question hash stored per entry → auto-invalidates if dataset is regenerated. |
| `config.py` | Default Gemini model bumped from `gemini-2.0-flash` (now blocked for new users) to `gemini-2.5-flash`. |
| `requirements.txt` | `google-generativeai>=0.4.0` → `google-genai>=1.0.0`. |
| `.env.example` | **NEW.** Template for `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`. |
| `.gitignore` | Added `.venv/`, `__pycache__/`, `*.pyc`, `chroma_db/`. |

### Local environment setup that's now needed

This branch will not run with just `pip install -r requirements.txt` on a system Python — the deps don't all have wheels for Python 3.14. Use Python 3.11 in a venv:

```bash
python3.11 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install torchvision   # silences a Streamlit watcher warning, not strictly required
cp .env.example .env                  # then fill in your keys
```

Ollama (optional, for local models):

```bash
sudo pacman -S ollama         # or your distro's equivalent
ollama serve &                # daemon must be running for the picker to see local models
ollama pull llama3            # add any models you want to compare
ollama pull mistral
```

### Schema notes for downstream branches

- Each entry in `results.json` is now a dict with these top-level keys (in addition to the prior metric fields): `provider`, `timestamp` (ISO-8601 UTC), `n_questions`, `t_runs`, `cache_hits`, `cache_misses`. Old entries from before this change are still readable — the leaderboard uses `.get(...)` with defaults and shows "—" for missing fields.
- `chroma_db/` is created automatically on first eval if absent. It is gitignored. To fully reset, delete the directory or click **🧹 Cache** in the GUI.
- `EvaluationResult` and `evaluate_model(...)` gained a `use_cache: bool = True` kwarg. Pass `False` to bypass cache without wiping it.

### Merging this branch into `main` / `pranika-feature` / others

Other branches that diverged from `main` before this work landed will have the old single-provider picker, the old `google-generativeai` SDK, and an overwriting `results.json`. To bring them in sync:

```bash
# from inside a clone, on the target branch:
git fetch origin
git merge origin/mihir
```

Expected conflicts and how to resolve them:

| File | Likely conflict | Resolution |
|---|---|---|
| `app.py` | Evaluate page rewritten end-to-end; Results page top bar added. | Take `mihir` version wholesale unless the other branch has unrelated UI work — in which case, port that work onto the new picker structure. |
| `pipeline.py` | `get_llm`, arg parsing, save logic all changed. | Take `mihir`. |
| `llm/gemini_llm.py` | Full rewrite (`google.generativeai` → `google.genai`). | Take `mihir`. Any other branch still on the deprecated SDK will hit `404` on Gemini 2.0 anyway. |
| `evaluation/evaluator.py` | New cache wiring + `cache_hits/misses` fields. | Take `mihir`. If another branch added metric logic, keep that and re-apply within the new cache flow. |
| `embeddings/encoder.py` | `_text_cache` LRU added. | Take `mihir`. |
| `storage/chroma_store.py` | API signature changed (added `question_text`, `reasoning_steps`, `raw_response`). | Take `mihir`. No other branch should be calling these — `ChromaStore` was previously dead code. |
| `requirements.txt` | `google-generativeai` → `google-genai`. | Take `mihir`. After merging, recreate the venv or `pip install -r requirements.txt && pip uninstall google-generativeai google-ai-generativelanguage`. |

After merging, anyone pulling onto a fresh checkout still needs to:
1. Rebuild their `.venv` from `requirements.txt` (Python 3.11).
2. Recreate their `.env` from `.env.example`.
3. Start `ollama serve` if they want local models.

### Known sharp edges

- **Gemini's `models.list()` returns models that aren't actually callable** for newly-created Google AI Studio accounts (e.g. `gemini-2.0-flash`). The picker shows them; the failure surfaces only at call time as a per-model `404 NOT_FOUND`. Other models in the same run are unaffected. Workaround: stick to `gemini-2.5-*`. A real pre-flight ping (1-token request per cloud model) would catch this but adds ~1s + a few cents per run; not yet implemented.
- Per-question metric scores are **not** cached (only LLM responses are). Recomputing scores from cached responses takes <1s per question and avoids invalidation issues when weights/runs change.
- **Sentence-transformers model load is the dominant startup cost** (~2–3 s per Python process). Streamlit re-uses the same process across runs, so this only hits once per session.

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

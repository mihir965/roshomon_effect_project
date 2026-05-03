# Slide deck — CS671 Team MVP

10-minute presentation. Each slide block has (a) what to put on screen and (b) speaker notes.
Plot files live under `presentation/plots/`.

---

## Slide 1 — Title (5 s)

**On screen**
- **Evaluating LLMs on Reasoning, Not Just Answers**
- *A framework for exposing the Rashomon Effect in language models*
- Team MVP — Varun, Pranika, Mihir
- CS671 · 2026-05-03

**Speaker notes**
> Hi, we're Team MVP. Our project asks a simple question — when two models give the same right answer, are they actually reasoning the same way? Spoiler: often, no.

---

## Slide 2 — Motivation and context (60 s) — *(Pranika)*

**On screen**
- LLM benchmarks today reward **final answers**: MMLU, GSM8K, BIG-bench, etc.
- Two models can land on the same correct answer via **completely different reasoning chains** — one principled, one accidental.
- The classic Rashomon Effect (Breiman, 2001): many models fit the data equally well but tell incompatible stories. Pick the wrong one, and you trust a brittle reasoner that happened to be right today.
- Real-world stakes: medical triage, legal analysis, autonomous decision systems — *how* a model arrives at an answer matters as much as *what* it answers.

**Speaker notes**
> Today's leaderboards collapse model quality into a single number — usually accuracy. But two models with identical accuracy can have wildly different internal reasoning. We call this the Rashomon Effect, after the 1950 Kurosawa film where four people give incompatible accounts of the same event. In high-stakes deployments, you don't just want a model that's right; you want one that's right for the right reasons. There is no widely-used framework that measures this. That's the gap we set out to fill.

---

## Slide 3 — Research/project question (30 s) — *(Pranika)*

**On screen**
> **Can we evaluate and recommend LLMs based on the *quality and alignment* of their reasoning chains, not just final-answer accuracy?**
>
> Concretely: define metrics that capture (a) full-chain alignment, (b) per-step coverage, (c) consistency under repeat sampling, (d) domain-knowledge grounding — and use them to surface Rashomon pairs.

**Speaker notes**
> Our research question, exact wording. The hypothesis is that adding reasoning-quality metrics on top of final-answer scoring will reveal Rashomon pairs that current benchmarks hide.

---

## Slide 4 — Methods and approach (120 s) — *(Varun)*

**On screen — left half (pipeline)**
```
HuggingFace dataset (50 hard-reasoning questions)
        ↓
Golden Truth JSON (steps + answer + must-include concepts)
        ↓
N models × T runs each, with CoT prompting
        ↓
Sentence-Transformer embeddings (all-MiniLM-L6-v2)
        ↓
Five reasoning metrics + composite FPS
        ↓
Streamlit dashboard + persistent ChromaDB cache
```

**On screen — right half (metrics table)**

| Metric | Formula | Catches |
|---|---|---|
| AAS | cos(E(model_ans), E(golden_ans)) | answer correctness |
| RAS | cos(E(model_chain), E(golden_chain)) | full-chain alignment |
| SLMS | mean over golden steps of max sim to any model step | step-level coverage |
| CS | 1 − Var(RAS₁…RAS_T) | consistency across runs |
| DKUS | \|Cg ∩ Cm\| / \|Cg\| | required concepts present |
| FPS | weighted sum of all 5 (sliders in UI) | composite recommendation |

**Speaker notes**
> Pipeline left, metrics right. Five metrics, each catching a different failure mode that a pure-accuracy score misses.
>
> The key design decision is **SLMS is asymmetric** — for every golden step, we ask "does the model cover this?" — not the other way. That penalizes missing logic without punishing models that show *more* work. CS uses T=3 repeated runs per question with temperature 0.7; if a model gives different reasoning chains for the same question across runs, its consistency score drops. DKUS is a simple concept-coverage check using substring matching on the must-include list.
>
> All five fold into FPS — Final Performance Score — with weights you can move live in the dashboard. We default to RAS=0.30, SLMS=0.25, AAS=0.20, DKUS=0.15, CS=0.10, putting reasoning above raw answer match.

---

## Slide 5 — Results: leaderboard (90 s) — *(Mihir)*

**On screen**
- Image: `plots/leaderboard.png`
- Caption: 7 models, default FPS weights
- Bullet: **Claude family dominates the composite score** — but Gemini-2.5-flash and gemini-3.1-pro-preview score lower despite being more recent / more expensive
- Bullet: **Mistral and llama3 (local, free) tie the cloud frontier on N=5** — local models are competitive on this slice

**Speaker notes**
> First headline result. We evaluated 7 models — 3 Anthropic, 2 Google, 2 Ollama-local — on hard-reasoning questions from a HuggingFace dataset. The composite score (FPS) ranks Claude haiku-4-5 and sonnet-4 at the top. The interesting finding isn't that Claude wins overall — it's the gap between Gemini's headline accuracy on standard benchmarks and how it scores when we include reasoning alignment.
>
> Caveat we'll come back to: the bars labeled N=5 and N=10 saw fewer questions than the N=20 ones. We'll talk about this in Limitations.

---

## Slide 6 — Results: the Rashomon Effect made visible (90 s) — *(Mihir)*

**On screen — left**
- Image: `plots/rashomon_scatter.png` (AAS vs RAS scatter)
- Caption: each light dot = (one question, one model). Big dots = model means.

**On screen — right (annotated example)**
> **Q_0005** — Crypto-trading indicator selection (RSI + TVA is correct).
>
> | Model | AAS | RAS | Answer |
> |---|---|---|---|
> | claude-haiku-4-5 | **0.84** | **0.75** | "RSI and TVA…" ✓ |
> | claude-sonnet-4 | **0.81** | **0.52** | "RSI and TVA…" ✓ |
>
> *Same family. Same right answer. **23-point gap** in reasoning alignment.*

**Speaker notes**
> This is the result the project was built to surface. Look at the scatter on the left — AAS on the X axis is "did you get the answer right," RAS on the Y axis is "did you reason your way there the same way the golden answer did." If accuracy were the whole story, dots would cluster on the diagonal. They don't — the spread above and below the diagonal *is* the Rashomon Effect.
>
> Concrete example on the right. Question 5 is a constraint-satisfaction problem about picking trading indicators. Both Claude haiku and Claude sonnet — same model family, mind you — pick the right answer. AAS scores within 3 points of each other. But sonnet's reasoning chain only matches golden at RAS=0.52, while haiku's matches at 0.75. They got there via different paths. If we'd ranked these on accuracy alone, we'd call them equivalent. They aren't.

---

## Slide 7 — Key takeaway (45 s) — *(Mihir)*

**On screen**
- Image: `plots/rank_flip.png` (left half)

> **Yes — reasoning quality and answer correctness are measurably distinct, and ranking models on accuracy alone hides the Rashomon Effect entirely.**
>
> - Per-question AAS vs RAS: **Pearson r = 0.30** across 90 (model × question) pairs — they barely correlate
> - Pure-accuracy ranking puts **mistral:7b at #1**; reasoning-aware ranking (FPS) puts it at **#4**, with Claude haiku jumping to the top
> - **6 of 21 pairwise rankings flip** (29 %) when we move from "rank by AAS" to "rank by FPS"
> - Rashomon pairs show up *within the same model family* — Claude haiku and sonnet, both correct on q_0005, RAS gap of 23 points
> - Framework runs in < 30 s per model on cached questions, so this can be a default deployment check, not a research afterthought

**Speaker notes**
> Key takeaway. Yes — reasoning quality is measurably distinct from accuracy, and including it in the ranking changes which model you'd recommend. The single most striking number on this slide: per-question AAS and RAS correlate at only Pearson 0.30 across our data. They are not the same thing.
>
> Practical implication: if you'd ranked our 7 models on raw accuracy, mistral 7B — a local, free model — would have come out on top. Once we include reasoning alignment, mistral falls four spots and Claude haiku takes #1. Six of twenty-one model-pair comparisons flip orientation between the two ranking methods. That's a third of all comparisons.
>
> And the Rashomon pairs aren't just across providers — they show up *within* the Claude family, between sibling models, which means even when you've committed to a vendor you still need this analysis.

---

## Slide 8 — Limitations and future work (60 s) — *(Varun)*

**On screen**
- **Dataset breadth**: only `hard-reasoning-en` from one HuggingFace dataset. 50 questions total, our results use up to 20.
- **Ragged N**: Ollama and Opus saw only 5–10 questions, not the full 20. Cache enables a uniform-N rerun in ~30 min — would strengthen the comparison.
- **Embedding choice**: `all-MiniLM-L6-v2` is small and fast but not domain-tuned. RAS scores would shift with a stronger encoder.
- **Golden truth quality**: dataset reasoning steps are gold-labeled by humans for German RAG; we use the English split, may have subtle artifacts.
- **DKUS is naive**: substring matching misses paraphrase. LLM-based concept verification is a clear upgrade.

**Future work**
- Cross-domain replication (math, code, biomedical Q&A)
- Encoder ablation: MPNet, BGE, or a fine-tuned reasoning encoder
- Per-step LLM judge for SLMS (currently embedding-based)
- Online recommendation: route a query to the best model for that *question's domain*

**Speaker notes**
> Five limitations, in honesty order. The biggest is dataset narrowness — we only ran one type of question. Second is the ragged N across models, which we know how to fix and just ran out of time for. Third is the embedding model — small, fast, but not the strongest available, so RAS numbers would shift if we used a domain-tuned encoder. Fourth, the golden truth comes from a German RAG dataset translated to English, which can introduce artifacts. Fifth, DKUS is the weakest metric — it's a substring match, so synonyms and paraphrases get missed.
>
> Future work, three concrete next steps: cross-domain replication, encoder ablation to confirm the rankings are robust, and an LLM-judge variant of SLMS for real semantic step-matching.

---

## Slide 9 — Thank you (5 s)

**On screen**
- "Thank you"
- "Questions?"

---

## Slide 10 — Team contributions (not presented; required)

**On screen**

| Member | Contributions |
|---|---|
| Varun | *(fill in)* |
| Pranika | *(fill in)* |
| Mihir | *(fill in)* |

---

## Slide 11 — GitHub repository (not presented; required)

**On screen**
- Repository: *(paste link here — e.g. `github.com/<user>/roshomon_effect_project`)*
- Branch: `mihir`
- Quick-start: `streamlit run app.py`

---

# Appendix slides

## A1 — Detailed metric formulas

**On screen**

```
AAS  = max(0, cos(E(answer_m), E(answer_g)))
RAS  = max(0, cos(E(chain_m), E(chain_g)))            chain = " ".join(steps)
SLMS = (1/|G|) Σ_{s∈G} max_{r∈M} cos(E(s), E(r))      asymmetric: golden→model
CS   = max(0, 1 − Var(RAS_1, …, RAS_T))               T = 3 in this run
DKUS = |Cg ∩ Cm| / |Cg|                                substring match, lowercased
FPS  = w_AAS·AAS + w_RAS·RAS + w_SLMS·SLMS + w_CS·CS + w_DKUS·DKUS
```

Default weights: w_AAS=0.20, w_RAS=0.30, w_SLMS=0.25, w_CS=0.10, w_DKUS=0.15.

---

## A2 — Per-metric breakdown across all models

**On screen**
- Image: `plots/radar.png`
- *Reading the chart*: shape = strengths profile. Claude family has near-flat radial coverage; Gemini falls off on AAS and DKUS.

---

## A3 — Per-metric grouped bars

**On screen**
- Image: `plots/metrics_grouped.png`
- All 7 models × 5 metrics laid out side-by-side for direct comparison.

---

## A4 — Reasoning consistency across repeated runs

**On screen**
- Image: `plots/cs_box.png`
- Caption: every model is ≥ 0.99 on CS — repeated runs of the same question produce nearly identical RAS.
- *Interpretation*: at temperature 0.7, models are surprisingly consistent at the chain level. Variance is in surface phrasing, not reasoning structure.

---

## A5 — Per-question fingerprint (5 questions all 7 models share)

**On screen**
- Image: `plots/per_question_heatmap.png`
- Each row is a model, each column is a question. Color is FPS for that (model, question) pair.
- Note q_0000 is the hardest: every model scores below 0.55. q_0004 is easiest. Difficulty is question-specific, not model-specific.

---

## A6 — Architecture and tech stack

**On screen**

```
Frontend     Streamlit + Plotly
LLM APIs     OpenAI · Anthropic · Google (google-genai) · Ollama (local)
Embeddings   sentence-transformers / all-MiniLM-L6-v2 (CUDA-accelerated)
Cache        ChromaDB persistent vector store, keyed by (qid, model, run)
Data         HuggingFace `avemio/German-RAG-ORPO-Alpaca-HESSIAN-AI` · hard-reasoning-en
```

- Cache invalidation: hash of question text stored per response → automatic re-query when dataset is rebuilt
- Cost: ~30 LLM calls per (model, 10 questions, T=3) run; cached re-runs free
- Local models compete on the same metrics — mistral:7b and llama3:8b ran on consumer GPU

---

## A7 — Live demo cheat-sheet (if asked)

**On screen / pocket card**

```
ollama serve &                          # one-time
.venv/bin/streamlit run app.py          # opens localhost:8501

GUI flow:
  Dataset    → Load / Build Golden Truth   (or skip if file exists)
  Evaluate   → pick models per provider → Start Evaluation
  Results    → leaderboard + radar + drill-down
              · 🗑 Clear all  → wipe leaderboard
              · 🧹 Cache (N)  → wipe LLM response cache
              · sidebar sliders → re-rank live without re-running
```

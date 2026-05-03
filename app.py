"""Streamlit evaluation dashboard."""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="LLM Reasoning Evaluation Framework",
    page_icon="🧠",
    layout="wide",
)

from config import DEFAULT_WEIGHTS, GOLDEN_TRUTH_PATH, T_RUNS  # noqa: E402


# ── helpers ───────────────────────────────────────────────────────────────────

def load_results_file(path: str = "results.json") -> list[dict] | None:
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def recompute_fps(results: list[dict], weights: dict) -> list[dict]:
    """Re-score FPS in-place using current sidebar weights (no re-query needed)."""
    for r in results:
        qrs = r.get("question_results", [])
        if not qrs:
            continue
        for qr in qrs:
            qr["fps"] = (
                weights["w_aas"] * qr["aas"]
                + weights["w_ras"] * qr["ras"]
                + weights["w_slms"] * qr["slms"]
                + weights["w_cs"] * qr["cs"]
                + weights["w_dkus"] * qr["dkus"]
            )
        r["mean_fps"] = sum(qr["fps"] for qr in qrs) / len(qrs)
    return results


def build_leaderboard(results: list[dict]) -> pd.DataFrame:
    rows = [{
        "Model": r["model_name"],
        "Provider": r.get("provider", "—"),
        "FPS":  round(r.get("mean_fps", 0), 4),
        "AAS":  round(r.get("mean_aas", 0), 4),
        "RAS":  round(r.get("mean_ras", 0), 4),
        "SLMS": round(r.get("mean_slms", 0), 4),
        "CS":   round(r.get("mean_cs", 0), 4),
        "DKUS": round(r.get("mean_dkus", 0), 4),
        "N":    r.get("n_questions", len(r.get("question_results", []))),
        "T":    r.get("t_runs", "—"),
        "Run at (UTC)": r.get("timestamp", "—"),
    } for r in results]
    df = pd.DataFrame(rows).sort_values("FPS", ascending=False).reset_index(drop=True)
    df.index += 1
    return df


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("🧠 LLM Eval Framework")
page = st.sidebar.radio("Navigation", ["Dataset", "Evaluate", "Results"])

st.sidebar.markdown("---")
st.sidebar.subheader("FPS Weights  (must sum to 1)")
w_aas  = st.sidebar.slider("AAS",  0.0, 1.0, DEFAULT_WEIGHTS["w_aas"],  0.05)
w_ras  = st.sidebar.slider("RAS",  0.0, 1.0, DEFAULT_WEIGHTS["w_ras"],  0.05)
w_slms = st.sidebar.slider("SLMS", 0.0, 1.0, DEFAULT_WEIGHTS["w_slms"], 0.05)
w_cs   = st.sidebar.slider("CS",   0.0, 1.0, DEFAULT_WEIGHTS["w_cs"],   0.05)
w_dkus = st.sidebar.slider("DKUS", 0.0, 1.0, DEFAULT_WEIGHTS["w_dkus"], 0.05)

weights = {"w_aas": w_aas, "w_ras": w_ras, "w_slms": w_slms, "w_cs": w_cs, "w_dkus": w_dkus}
total_w = sum(weights.values())
if abs(total_w - 1.0) > 0.01:
    st.sidebar.error(f"Weights sum to {total_w:.2f} — adjust to equal 1.0")

METRICS = ["AAS", "RAS", "SLMS", "CS", "DKUS"]


# ── Dataset page ──────────────────────────────────────────────────────────────

if page == "Dataset":
    st.title("Dataset")
    st.caption("Source: `avemio/German-RAG-ORPO-Alpaca-HESSIAN-AI`  |  split: `hard-reasoning-en`")

    col1, col2 = st.columns([3, 1])
    with col1:
        n_samples = st.number_input("Questions to load", min_value=5, max_value=500, value=20, step=5)
    with col2:
        force_rebuild = st.checkbox("Force rebuild")

    if st.button("Load / Build Golden Truth"):
        with st.spinner("Fetching from HuggingFace…"):
            try:
                from data.dataset_loader import (
                    load_hf_dataset, dataset_to_golden_truth, save_golden_truth
                )
                ds = load_hf_dataset(n_samples=n_samples)
                st.success(f"Loaded {len(ds)} rows  |  columns: `{ds.column_names}`")

                if force_rebuild or not Path(GOLDEN_TRUTH_PATH).exists():
                    records = dataset_to_golden_truth(ds, n_samples=n_samples)
                    save_golden_truth(records)
                    st.success(f"Built {len(records)} golden truth records → `{GOLDEN_TRUTH_PATH}`")
            except Exception as exc:
                st.error(f"Error: {exc}")

    if Path(GOLDEN_TRUTH_PATH).exists():
        from data.dataset_loader import load_golden_truth
        records = load_golden_truth()
        st.subheader(f"Golden Truth  ({len(records)} records)")

        idx = st.slider("Record index", 0, max(0, len(records) - 1), 0)
        rec = records[idx]

        left, right = st.columns(2)
        with left:
            st.markdown(
                f"**ID:** `{rec['question_id']}`  ·  **Domain:** {rec['task_domain']}  ·  "
                f"**Difficulty:** {rec['difficulty_level']}"
            )
            st.markdown("**Question**")
            st.info(rec["question"])
            concepts = rec["evaluation_criteria"]["must_include_concepts"]
            if concepts:
                st.markdown(f"**Key concepts:** {', '.join(concepts[:8])}")
        with right:
            st.markdown("**Golden Reasoning Steps**")
            for s in rec["golden_reasoning_steps"]:
                st.markdown(f"**{s['step_number']}.** {s['description']}")
            st.success(f"**Answer:** {rec['golden_final_answer']}")
    else:
        st.info("No golden truth file found — click the button above to build one.")


# ── Evaluate page ─────────────────────────────────────────────────────────────

elif page == "Evaluate":
    st.title("Run Evaluation")

    if not Path(GOLDEN_TRUTH_PATH).exists():
        st.warning("Build the golden truth dataset first (Dataset page).")
        st.stop()

    from datetime import datetime, timezone  # noqa: E402
    from llm.model_registry import all_available_models, parse_model_spec  # noqa: E402

    @st.cache_data(ttl=120, show_spinner="Discovering available models…")
    def _discover():
        return all_available_models()

    refresh_col, _ = st.columns([1, 5])
    with refresh_col:
        if st.button("🔄 Refresh model list"):
            _discover.clear()

    catalog = _discover()

    st.markdown(
        "Pick any models you want to compare — multiple per provider is fine. "
        "The list is fetched live from each provider; an empty section means no API key, "
        "no Ollama daemon, or the network call failed."
    )

    selected_specs: list[str] = []
    cols = st.columns(4)
    for col, provider in zip(cols, ("openai", "anthropic", "gemini", "ollama")):
        with col:
            models = catalog.get(provider, [])
            label = f"{provider}  ({len(models)})"
            if not models:
                st.markdown(f"**{label}**")
                if provider == "openai":
                    st.caption("No `OPENAI_API_KEY`")
                elif provider == "anthropic":
                    st.caption("No `ANTHROPIC_API_KEY`")
                elif provider == "gemini":
                    st.caption("No `GOOGLE_API_KEY`")
                else:
                    st.caption("Ollama daemon not reachable")
                continue
            picks = st.multiselect(label, models, key=f"pick_{provider}")
            selected_specs.extend(f"{provider}:{m}" for m in picks)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        n_q = st.number_input("Questions", min_value=1, max_value=200, value=10, step=5)
    with col2:
        t_runs = st.number_input("Runs / question (CS)", min_value=1, max_value=10, value=T_RUNS)

    if abs(total_w - 1.0) > 0.01:
        st.error("Fix sidebar weights to sum to 1.0 before running.")
        st.stop()

    if not selected_specs:
        st.info("Pick at least one model above.")
        st.stop()

    st.write(f"**Selected ({len(selected_specs)})**: " + ", ".join(f"`{s}`" for s in selected_specs))

    if st.button("Start Evaluation", type="primary"):
        from data.dataset_loader import load_golden_truth
        from evaluation.evaluator import evaluate_model, results_to_dict

        golden = load_golden_truth()[:n_q]
        new_runs: list[dict] = []
        progress = st.progress(0.0)
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

        for i, spec in enumerate(selected_specs):
            provider, model_id = parse_model_spec(spec)
            st.write(f"Querying **{spec}**…")
            try:
                if provider == "openai":
                    from llm.openai_llm import OpenAILLM
                    llm = OpenAILLM(model_name=model_id)
                elif provider == "anthropic":
                    from llm.anthropic_llm import AnthropicLLM
                    llm = AnthropicLLM(model_name=model_id)
                elif provider == "gemini":
                    from llm.gemini_llm import GeminiLLM
                    llm = GeminiLLM(model_name=model_id)
                else:
                    from llm.ollama_llm import OllamaLLM
                    llm = OllamaLLM(model_name=model_id)

                result = evaluate_model(llm, golden, weights, t_runs)
                d = results_to_dict(result)
                d["provider"] = provider
                d["timestamp"] = timestamp
                d["n_questions"] = len(golden)
                d["t_runs"] = t_runs
                new_runs.append(d)
                st.success(
                    f"{spec} — FPS: {result.mean_fps:.3f} | "
                    f"AAS: {result.mean_aas:.3f} | RAS: {result.mean_ras:.3f} | "
                    f"SLMS: {result.mean_slms:.3f} | CS: {result.mean_cs:.3f} | "
                    f"DKUS: {result.mean_dkus:.3f}  ·  "
                    f"cache: {result.cache_hits} hits / {result.cache_misses} misses"
                )
            except Exception as exc:
                st.error(f"{spec} failed: {exc}")

            progress.progress((i + 1) / len(selected_specs))

        if new_runs:
            results_path = Path("results.json")
            existing = []
            if results_path.exists():
                try:
                    with open(results_path) as f:
                        existing = json.load(f)
                except Exception:
                    existing = []

            new_names = {r["model_name"] for r in new_runs}
            kept = [r for r in existing if r["model_name"] not in new_names]
            merged = kept + new_runs

            with open(results_path, "w") as f:
                json.dump(merged, f, indent=2)
            st.session_state["results"] = merged
            st.balloons()
            replaced = len(existing) - len(kept)
            st.info(
                f"Saved **{len(new_runs)}** new run(s); replaced **{replaced}** prior entry by name; "
                f"**{len(merged)}** total models in `results.json`. Navigate to the Results page."
            )


# ── Results page ──────────────────────────────────────────────────────────────

elif page == "Results":
    st.title("Results")

    results = st.session_state.get("results") or load_results_file()
    if not results:
        st.info("No results yet — run an evaluation first.")
        st.stop()

    # Top-bar controls: drop a model, wipe leaderboard, or wipe LLM response cache
    top_l, top_m, top_r1, top_r2 = st.columns([3, 2, 1, 1])
    with top_l:
        model_names_all = [r["model_name"] for r in results]
        to_drop = st.multiselect("Drop specific models", model_names_all, key="drop_models")
    with top_m:
        if to_drop and st.button("Drop selected"):
            kept = [r for r in results if r["model_name"] not in set(to_drop)]
            with open("results.json", "w") as f:
                json.dump(kept, f, indent=2)
            st.session_state["results"] = kept
            st.rerun()
    with top_r1:
        if st.button("🗑 Clear all", help="Wipe results.json"):
            Path("results.json").unlink(missing_ok=True)
            st.session_state.pop("results", None)
            st.rerun()
    with top_r2:
        try:
            from storage.chroma_store import ChromaStore
            _store = ChromaStore()
            cache_n = _store.count()
        except Exception:
            cache_n = 0
        if st.button(f"🧹 Cache ({cache_n})", help="Wipe cached LLM responses (chroma_db/)"):
            try:
                ChromaStore().wipe()
                st.success("LLM response cache cleared.")
            except Exception as exc:
                st.error(f"Failed to wipe cache: {exc}")
            st.rerun()

    results = recompute_fps(results, weights)
    df = build_leaderboard(results)

    # Leaderboard
    st.subheader("Leaderboard")
    st.dataframe(df, width="stretch")
    winner = df.iloc[0]["Model"]
    st.success(f"Recommended model: **{winner}**  (FPS = {df.iloc[0]['FPS']})")

    # Radar chart
    fig_radar = go.Figure()
    for _, row in df.iterrows():
        vals = [row[m] for m in METRICS]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=METRICS + [METRICS[0]],
            fill="toself",
            name=row["Model"],
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Metric Radar",
        height=450,
    )
    st.plotly_chart(fig_radar, width="stretch")

    # Grouped bar
    fig_bar = px.bar(
        df.melt(id_vars="Model", value_vars=METRICS),
        x="variable", y="value", color="Model",
        barmode="group",
        labels={"variable": "Metric", "value": "Score"},
        title="Score Breakdown by Metric",
    )
    st.plotly_chart(fig_bar, width="stretch")

    # Per-question drill-down
    st.subheader("Question Drill-Down")
    model_names = [r["model_name"] for r in results]
    sel_model = st.selectbox("Model", model_names)
    model_result = next(r for r in results if r["model_name"] == sel_model)

    qr_rows = [{
        "Question ID": qr["question_id"],
        "AAS":  round(qr["aas"], 3),
        "RAS":  round(qr["ras"], 3),
        "SLMS": round(qr["slms"], 3),
        "CS":   round(qr["cs"], 3),
        "DKUS": round(qr["dkus"], 3),
        "FPS":  round(qr.get("fps", 0), 3),
    } for qr in model_result["question_results"]]
    st.dataframe(pd.DataFrame(qr_rows), width="stretch")

    # Question detail
    qids = [qr["question_id"] for qr in model_result["question_results"]]
    sel_qid = st.selectbox("Question detail", qids)
    qr = next(qr for qr in model_result["question_results"] if qr["question_id"] == sel_qid)

    if Path(GOLDEN_TRUTH_PATH).exists():
        from data.dataset_loader import load_golden_truth
        golden_map = {g["question_id"]: g for g in load_golden_truth()}
        gt = golden_map.get(sel_qid)
        if gt:
            left, right = st.columns(2)
            with left:
                st.markdown("**Question**")
                st.info(gt["question"])
                st.markdown("**Golden Steps**")
                for s in gt["golden_reasoning_steps"]:
                    st.markdown(f"**{s['step_number']}.** {s['description']}")
                st.success(f"**Golden Answer:** {gt['golden_final_answer']}")
            with right:
                runs = qr.get("runs", [])
                if runs:
                    first = runs[0]
                    st.markdown("**Model Steps**")
                    for i, step in enumerate(first.get("reasoning_steps", []), 1):
                        st.markdown(f"**{i}.** {step}")
                    st.info(f"**Model Answer:** {first.get('final_answer', 'N/A')}")
                    if len(runs) > 1:
                        with st.expander(f"All {len(runs)} runs (for CS)"):
                            for run in runs:
                                st.markdown(f"**Run {run['run_index']}:** {run.get('final_answer')}")

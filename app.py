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

from config import DEFAULT_WEIGHTS, GOLDEN_TRUTH_PATH, T_RUNS, HF_MODELS  # noqa: E402


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
        "FPS":  round(r.get("mean_fps", 0), 4),
        "AAS":  round(r.get("mean_aas", 0), 4),
        "RAS":  round(r.get("mean_ras", 0), 4),
        "SLMS": round(r.get("mean_slms", 0), 4),
        "CS":   round(r.get("mean_cs", 0), 4),
        "DKUS": round(r.get("mean_dkus", 0), 4),
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

    col1, col2, col3 = st.columns(3)
    with col1:
        all_model_choices = ["openai", "anthropic", "gemini", "llama", "qwen", "llama4", *HF_MODELS.keys()]
        selected_models = st.multiselect(
            "LLMs to evaluate", all_model_choices, default=["openai"]
        )
    with col2:
        n_q = st.number_input("Questions", min_value=1, max_value=200, value=5, step=5)
    with col3:
        t_runs = st.number_input("Runs / question (CS)", min_value=1, max_value=10, value=T_RUNS)
        if t_runs == 1:
            st.caption("CS will be 1.0 for all models (need ≥2 runs for a real variance signal)")

    if abs(total_w - 1.0) > 0.01:
        st.error("Fix sidebar weights to sum to 1.0 before running.")
        st.stop()

    if not selected_models:
        st.warning("Select at least one model.")
        st.stop()

    if st.button("Start Evaluation", type="primary"):
        from data.dataset_loader import load_golden_truth
        from evaluation.evaluator import evaluate_model, results_to_dict

        golden = load_golden_truth()[:n_q]
        all_results = []
        progress = st.progress(0.0)

        for i, model_name in enumerate(selected_models):
            st.write(f"Querying **{model_name}**…")
            try:
                if model_name in ("llama", "qwen", "llama4"):
                    from llm.groq_llm import GroqLLM
                    llm = GroqLLM(model_name)
                elif model_name == "openai":
                    from llm.openai_llm import OpenAILLM
                    llm = OpenAILLM()
                elif model_name == "anthropic":
                    from llm.anthropic_llm import AnthropicLLM
                    llm = AnthropicLLM()
                elif model_name == "gemini":
                    from llm.gemini_llm import GeminiLLM
                    llm = GeminiLLM()
                else:
                    from llm.hf_llm import HuggingFaceLLM
                    llm = HuggingFaceLLM(HF_MODELS[model_name])

                result = evaluate_model(llm, golden, weights, t_runs)
                d = results_to_dict(result)
                all_results.append(d)
                st.success(
                    f"{model_name} — FPS: {result.mean_fps:.3f} | "
                    f"AAS: {result.mean_aas:.3f} | RAS: {result.mean_ras:.3f} | "
                    f"SLMS: {result.mean_slms:.3f} | CS: {result.mean_cs:.3f} | "
                    f"DKUS: {result.mean_dkus:.3f}"
                )
            except Exception as exc:
                st.error(f"{model_name} failed: {exc}")

            progress.progress((i + 1) / len(selected_models))

        if all_results:
            with open("results.json", "w") as f:
                json.dump(all_results, f, indent=2)
            st.session_state["results"] = all_results
            st.balloons()
            st.info("Results saved to `results.json`. Navigate to the Results page.")


# ── Results page ──────────────────────────────────────────────────────────────

elif page == "Results":
    st.title("Results")

    results = st.session_state.get("results") or load_results_file()
    if not results:
        st.info("No results yet — run an evaluation first.")
        st.stop()

    results = recompute_fps(results, weights)
    df = build_leaderboard(results)

    # ── Leaderboard ───────────────────────────────────────────────────────────
    st.subheader("Leaderboard")
    st.dataframe(df, use_container_width=True)
    winner = df.iloc[0]["Model"]
    st.success(f"Recommended model: **{winner}**  (FPS = {df.iloc[0]['FPS']})")

    # ── Overview charts (radar + grouped bar) ─────────────────────────────────
    st.subheader("Overall Comparison")
    col_radar, col_bar = st.columns(2)

    with col_radar:
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
            height=400,
            margin=dict(t=50, b=20),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_bar:
        fig_bar = px.bar(
            df.melt(id_vars="Model", value_vars=METRICS),
            x="variable", y="value", color="Model",
            barmode="group",
            labels={"variable": "Metric", "value": "Score"},
            title="Score Breakdown by Metric",
            height=400,
        )
        fig_bar.update_layout(margin=dict(t=50, b=20), yaxis_range=[0, 1])
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── FPS composite bar ─────────────────────────────────────────────────────
    st.subheader("Final Performance Score (FPS)")
    fig_fps = px.bar(
        df.sort_values("FPS", ascending=True),
        x="FPS", y="Model", orientation="h",
        color="Model",
        text="FPS",
        title="Composite FPS (weighted sum of all metrics)",
        height=200 + 60 * len(df),
    )
    fig_fps.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig_fps.update_layout(xaxis_range=[0, 1], showlegend=False, margin=dict(t=50))
    st.plotly_chart(fig_fps, use_container_width=True)

    # ── Per-question line charts ───────────────────────────────────────────────
    st.subheader("Per-Question Performance")
    st.caption("How each model scores on every individual question — good for spotting where one model beats the other.")

    metric_choice = st.selectbox("Metric to plot per question", METRICS + ["FPS"], index=1)

    # Build a long-form dataframe of per-question scores
    pq_rows = []
    for r in results:
        for qr in r.get("question_results", []):
            pq_rows.append({
                "Model": r["model_name"],
                "Question": qr["question_id"],
                "AAS":  qr["aas"],
                "RAS":  qr["ras"],
                "SLMS": qr["slms"],
                "CS":   qr["cs"],
                "DKUS": qr["dkus"],
                "FPS":  qr.get("fps", 0),
            })
    pq_df = pd.DataFrame(pq_rows)

    fig_pq = px.line(
        pq_df, x="Question", y=metric_choice, color="Model",
        markers=True,
        title=f"{metric_choice} per Question",
        labels={"Question": "Question ID", metric_choice: metric_choice},
    )
    fig_pq.update_layout(yaxis_range=[0, 1], xaxis_tickangle=-45)
    st.plotly_chart(fig_pq, use_container_width=True)

    # ── Score distribution box plots ──────────────────────────────────────────
    st.subheader("Score Distributions")
    st.caption("Box plots show median, spread, and outliers across all questions.")

    fig_box = px.box(
        pq_df.melt(id_vars="Model", value_vars=METRICS, var_name="Metric", value_name="Score"),
        x="Metric", y="Score", color="Model",
        points="all",
        title="Distribution of Scores Across Questions",
    )
    fig_box.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig_box, use_container_width=True)

    # ── Head-to-head delta (only when exactly 2 models) ───────────────────────
    if len(results) == 2:
        st.subheader("Head-to-Head: Score Delta per Question")
        st.caption("Positive = first model wins on that question; negative = second model wins.")

        m1_name, m2_name = results[0]["model_name"], results[1]["model_name"]
        m1_df = pq_df[pq_df["Model"] == m1_name].set_index("Question")
        m2_df = pq_df[pq_df["Model"] == m2_name].set_index("Question")
        common_qs = m1_df.index.intersection(m2_df.index)

        delta_metric = st.selectbox("Metric for delta", METRICS + ["FPS"], index=1, key="delta_metric")
        delta = (m1_df.loc[common_qs, delta_metric] - m2_df.loc[common_qs, delta_metric]).reset_index()
        delta.columns = ["Question", "Delta"]
        delta["Winner"] = delta["Delta"].apply(lambda v: m1_name if v > 0 else m2_name)

        fig_delta = px.bar(
            delta, x="Question", y="Delta", color="Winner",
            title=f"{delta_metric} Delta  ({m1_name} − {m2_name})",
            labels={"Delta": f"Δ {delta_metric}"},
        )
        fig_delta.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_delta.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_delta, use_container_width=True)

    # ── Per-question drill-down ────────────────────────────────────────────────
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
    st.dataframe(pd.DataFrame(qr_rows), use_container_width=True)

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

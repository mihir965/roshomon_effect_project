"""Side-by-side reasoning comparison for two models on a single question.

Run with:  streamlit run compare.py
Reads results.json produced by pipeline.py.
"""

import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Reasoning Comparison",
    page_icon="🔍",
    layout="wide",
)

RESULTS_FILE = "results.json"
GOLDEN_FILE  = "data/golden_truth.json"

st.title("Reasoning Comparison — Rashomon Effect Explorer")
st.caption("Pick two models and a question to see how they reasoned their way to the same (or different) answer.")

# ── Load data ─────────────────────────────────────────────────────────────────

if not Path(RESULTS_FILE).exists():
    st.error(f"`{RESULTS_FILE}` not found. Run `pipeline.py` first.")
    st.stop()

with open(RESULTS_FILE) as f:
    all_results: list[dict] = json.load(f)

golden_map: dict[str, dict] = {}
if Path(GOLDEN_FILE).exists():
    with open(GOLDEN_FILE) as f:
        golden_map = {g["question_id"]: g for g in json.load(f)}

model_names = [r["model_name"] for r in all_results]

if len(model_names) < 2:
    st.error("Need at least 2 models in results.json. Run the pipeline with 2+ models.")
    st.stop()

# ── Model & question selectors ────────────────────────────────────────────────

col_a, col_b, col_q = st.columns([2, 2, 3])

with col_a:
    model_a_name = st.selectbox("Model A", model_names, index=0)
with col_b:
    remaining = [m for m in model_names if m != model_a_name]
    model_b_name = st.selectbox("Model B", remaining, index=0)
with col_q:
    result_a = next(r for r in all_results if r["model_name"] == model_a_name)
    question_ids = [qr["question_id"] for qr in result_a["question_results"]]
    selected_qid = st.selectbox("Question", question_ids)

result_b = next(r for r in all_results if r["model_name"] == model_b_name)
qr_a = next(qr for qr in result_a["question_results"] if qr["question_id"] == selected_qid)
qr_b = next((qr for qr in result_b["question_results"] if qr["question_id"] == selected_qid), None)

st.markdown("---")

# ── Question & golden answer ───────────────────────────────────────────────────

gt = golden_map.get(selected_qid)
if gt:
    st.subheader(f"Question  `{selected_qid}`")
    st.info(gt["question"])

    with st.expander("Golden reference answer & reasoning steps"):
        gcol1, gcol2 = st.columns(2)
        with gcol1:
            st.markdown("**Golden Reasoning Steps**")
            for s in gt["golden_reasoning_steps"]:
                st.markdown(f"**{s['step_number']}.** {s['description']}")
        with gcol2:
            st.success(f"**Golden Answer:** {gt['golden_final_answer']}")

st.markdown("---")

# ── Side-by-side reasoning ────────────────────────────────────────────────────

run_a = qr_a["runs"][0]
run_b = (qr_b["runs"][0] if qr_b else None)

answer_a = run_a["final_answer"]
answer_b = run_b["final_answer"] if run_b else "N/A"
answers_match = answer_a.strip().lower() == answer_b.strip().lower()

# Answer agreement banner
if answers_match:
    st.success(f"Both models gave the **same final answer**: _{answer_a}_")
else:
    st.warning(
        f"Models gave **different answers**  |  "
        f"**{model_a_name.split('/')[-1]}**: _{answer_a}_   "
        f"**{model_b_name.split('/')[-1]}**: _{answer_b}_"
    )

st.markdown("### Reasoning Chains")
left, right = st.columns(2)

with left:
    st.markdown(f"#### Model A — `{model_a_name.split('/')[-1]}`")
    steps_a = run_a["reasoning_steps"]
    if steps_a:
        for i, step in enumerate(steps_a, 1):
            st.markdown(f"**Step {i}:** {step}")
    else:
        st.markdown(run_a["raw_response"])
    st.markdown(f"**Final answer:** {answer_a}")

with right:
    st.markdown(f"#### Model B — `{model_b_name.split('/')[-1]}`")
    if run_b:
        steps_b = run_b["reasoning_steps"]
        if steps_b:
            for i, step in enumerate(steps_b, 1):
                st.markdown(f"**Step {i}:** {step}")
        else:
            st.markdown(run_b["raw_response"])
        st.markdown(f"**Final answer:** {answer_b}")
    else:
        st.warning("No result for this model on this question.")

st.markdown("---")

# ── Scores comparison ─────────────────────────────────────────────────────────

st.markdown("### Metric Scores for This Question")

METRICS = ["AAS", "RAS", "SLMS", "CS", "DKUS", "FPS"]
scores_a = [qr_a["aas"], qr_a["ras"], qr_a["slms"], qr_a["cs"], qr_a["dkus"], qr_a["fps"]]
scores_b = [qr_b["aas"], qr_b["ras"], qr_b["slms"], qr_b["cs"], qr_b["dkus"], qr_b["fps"]] if qr_b else [0] * 6

mcol1, mcol2 = st.columns(2)

# Score table
with mcol1:
    import pandas as pd
    score_df = pd.DataFrame({
        "Metric": METRICS,
        model_a_name.split("/")[-1]: [round(s, 4) for s in scores_a],
        model_b_name.split("/")[-1]: [round(s, 4) for s in scores_b],
        "Winner": [
            model_a_name.split("/")[-1] if a > b else (model_b_name.split("/")[-1] if b > a else "Tie")
            for a, b in zip(scores_a, scores_b)
        ],
    })
    st.dataframe(score_df, use_container_width=True, hide_index=True)

# Radar chart
with mcol2:
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores_a + [scores_a[0]],
        theta=METRICS + [METRICS[0]],
        fill="toself",
        name=model_a_name.split("/")[-1],
    ))
    fig.add_trace(go.Scatterpolar(
        r=scores_b + [scores_b[0]],
        theta=METRICS + [METRICS[0]],
        fill="toself",
        name=model_b_name.split("/")[-1],
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=f"Radar — {selected_qid}",
        height=380,
        margin=dict(t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

# RAS callout — the key Rashomon metric
st.markdown("---")
ras_a, ras_b = qr_a["ras"], (qr_b["ras"] if qr_b else 0)
ras_delta = abs(ras_a - ras_b)

st.markdown("### Reasoning Alignment Score (RAS) — The Rashomon Metric")
st.caption(
    "RAS measures how closely a model's reasoning chain matches the golden reference. "
    "Two models can reach the same answer via very different reasoning paths — that's the Rashomon Effect."
)

rcol1, rcol2, rcol3 = st.columns(3)
rcol1.metric(f"RAS — {model_a_name.split('/')[-1]}", f"{ras_a:.4f}")
rcol2.metric(f"RAS — {model_b_name.split('/')[-1]}", f"{ras_b:.4f}")
rcol3.metric("RAS Delta", f"{ras_delta:.4f}",
             help="How differently the two models reasoned. High delta = strong Rashomon Effect on this question.")

if ras_delta > 0.15:
    st.error(
        f"Large RAS gap ({ras_delta:.3f}) — strong Rashomon Effect: "
        "these models reached their answers via very different reasoning paths."
    )
elif ras_delta > 0.05:
    st.warning(f"Moderate RAS gap ({ras_delta:.3f}) — reasoning paths diverge somewhat.")
else:
    st.success(f"Small RAS gap ({ras_delta:.3f}) — models reasoned similarly on this question.")

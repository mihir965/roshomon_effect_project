"""Static plots for the slide deck + report.

Run: .venv/bin/python presentation/generate_plots.py
Outputs PNGs into presentation/plots/.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "plots"
OUT.mkdir(exist_ok=True)

with open(ROOT / "results.json") as f:
    results = json.load(f)
with open(ROOT / "data" / "golden_truth.json") as f:
    golden = json.load(f)
gmap = {g["question_id"]: g for g in golden}

METRICS = ["AAS", "RAS", "SLMS", "CS", "DKUS"]
METRIC_KEYS = ["mean_aas", "mean_ras", "mean_slms", "mean_cs", "mean_dkus"]

# Sort models by FPS desc
results = sorted(results, key=lambda r: r.get("mean_fps", 0), reverse=True)

# Short display names
def short(name: str) -> str:
    name = name.replace(":latest", "")
    name = name.replace("-20251001", "")
    name = name.replace("-20251101", "")
    name = name.replace("-20250514", "")
    name = name.replace("-preview", "")
    return name

names = [short(r["model_name"]) for r in results]
n_per = [r.get("n_questions", len(r.get("question_results", []))) for r in results]

PROVIDER_COLOR = {
    "anthropic": "#cc785c",
    "gemini": "#4285f4",
    "openai": "#10a37f",
    "ollama": "#7c3aed",
}
colors = [PROVIDER_COLOR.get(r.get("provider", ""), "#888") for r in results]

# ─────────────────────────────────────────────────────────────────────────────
# 1. FPS leaderboard bar chart
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.5))
fps_vals = [r.get("mean_fps", 0) for r in results]
bars = ax.barh(names[::-1], fps_vals[::-1], color=colors[::-1], edgecolor="black", linewidth=0.5)
for bar, n, fps in zip(bars, n_per[::-1], fps_vals[::-1]):
    ax.text(fps + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{fps:.3f}  (N={n})", va="center", fontsize=9)
ax.set_xlim(0, max(fps_vals) * 1.25)
ax.set_xlabel("Final Performance Score (FPS)")
ax.set_title("Model leaderboard — composite score across 5 reasoning metrics")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(OUT / "leaderboard.png", dpi=160, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 2. Radar chart — metric breakdown
# ─────────────────────────────────────────────────────────────────────────────
angles = np.linspace(0, 2 * np.pi, len(METRICS), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
for r, name, color in zip(results, names, colors):
    vals = [r.get(k, 0) for k in METRIC_KEYS]
    vals += vals[:1]
    ax.plot(angles, vals, color=color, linewidth=1.8, label=name)
    ax.fill(angles, vals, color=color, alpha=0.07)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(METRICS, fontsize=11)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_ylim(0, 1)
ax.set_title("Per-metric breakdown — every model has uneven strengths", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.05), fontsize=9)
plt.tight_layout()
plt.savefig(OUT / "radar.png", dpi=160, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 3. Grouped bar — all five metrics side by side
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(METRICS))
width = 0.10
for i, (r, name, color) in enumerate(zip(results, names, colors)):
    vals = [r.get(k, 0) for k in METRIC_KEYS]
    offset = (i - len(results) / 2) * width + width / 2
    ax.bar(x + offset, vals, width, label=name, color=color, edgecolor="black", linewidth=0.3)
ax.set_xticks(x)
ax.set_xticklabels(METRICS, fontsize=11)
ax.set_ylabel("Score (0 – 1)")
ax.set_title("Per-metric scores by model")
ax.legend(loc="upper right", fontsize=8, ncol=2)
ax.set_ylim(0, 1.1)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(OUT / "metrics_grouped.png", dpi=160, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 4. AAS vs RAS scatter — the Rashomon plot
# ─────────────────────────────────────────────────────────────────────────────
# AAS = answer correctness; RAS = reasoning alignment.
# Rashomon Effect = high AAS / lower RAS = right answer, different reasoning.
# Plot each MODEL's mean (AAS, RAS) AND each per-question (aas, ras) for the N=20 models
fig, ax = plt.subplots(figsize=(8.5, 6))

# Per-question dots (light)
for r, name, color in zip(results, names, colors):
    if r.get("n_questions", 0) < 20:
        continue
    qrs = r.get("question_results", [])
    aas = [qr["aas"] for qr in qrs]
    ras = [qr["ras"] for qr in qrs]
    ax.scatter(aas, ras, color=color, alpha=0.30, s=22)

# Model means (dark, labelled)
for r, name, color in zip(results, names, colors):
    ax.scatter(r.get("mean_aas", 0), r.get("mean_ras", 0),
               color=color, edgecolor="black", s=160, zorder=5)
    ax.annotate(name,
                xy=(r["mean_aas"], r["mean_ras"]),
                xytext=(8, 4), textcoords="offset points",
                fontsize=9)

# Diagonal reference
lims = [0, 1]
ax.plot(lims, lims, "--", color="grey", alpha=0.4, linewidth=1)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("AAS (answer correctness)")
ax.set_ylabel("RAS (full-chain reasoning alignment)")
ax.set_title("Rashomon plot — same answer, different reasoning\n"
             "Light dots: per-question scores (N=20 models). Big dots: model means.")
ax.grid(True, alpha=0.25)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(OUT / "rashomon_scatter.png", dpi=160, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 5. CS distribution box plot — consistency across runs
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.5))
data, labels, colors_box = [], [], []
for r, name, color in zip(results, names, colors):
    qrs = r.get("question_results", [])
    if not qrs:
        continue
    data.append([qr["cs"] for qr in qrs])
    labels.append(name)
    colors_box.append(color)

bp = ax.boxplot(data, labels=labels, patch_artist=True, vert=True, widths=0.55, showfliers=True)
for patch, color in zip(bp["boxes"], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
for median in bp["medians"]:
    median.set(color="black", linewidth=1.5)
ax.set_ylabel("Consistency Score (1 − Var(RAS) across T=3 runs)")
ax.set_title("Reasoning stability across repeated runs of the same question")
ax.set_ylim(0.96, 1.005)
plt.xticks(rotation=20, ha="right")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(OUT / "cs_box.png", dpi=160, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 6. Per-question agreement heatmap (5 questions all 7 models share)
# ─────────────────────────────────────────────────────────────────────────────
qids_by_model = {r["model_name"]: {qr["question_id"]: qr for qr in r["question_results"]} for r in results}
all_qids = sorted(set.intersection(*[set(d.keys()) for d in qids_by_model.values()]))

if all_qids:
    fig, ax = plt.subplots(figsize=(min(2 + len(all_qids) * 0.9, 12), 4.5))
    matrix = np.array([
        [qids_by_model[r["model_name"]][q]["fps"] for q in all_qids]
        for r in results
    ])
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(all_qids)))
    ax.set_xticklabels(all_qids, rotation=30, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="black" if matrix[i, j] > 0.5 else "white")
    ax.set_title(f"Per-question FPS — {len(all_qids)} questions seen by every model")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="FPS")
    plt.tight_layout()
    plt.savefig(OUT / "per_question_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 7. Rank flip: AAS-only ranking vs FPS ranking
# ─────────────────────────────────────────────────────────────────────────────
results_for_rank = sorted(json.load(open(ROOT / "results.json")), key=lambda r: -r.get("mean_fps", 0))
aas_rank = sorted(results_for_rank, key=lambda r: -r["mean_aas"])
fps_rank = results_for_rank  # already sorted by FPS desc

aas_order = [r["model_name"] for r in aas_rank]
fps_order = [r["model_name"] for r in fps_rank]

fig, ax = plt.subplots(figsize=(9, 6.5))
n = len(aas_order)
ax.set_xlim(0, 3)
ax.set_ylim(n - 0.5, -2.5)  # invert + room for headers
ax.axis("off")

ax.text(0.5, -1.6, "Rank by AAS\n(answer-only)", ha="center", fontsize=12, fontweight="bold")
ax.text(2.5, -1.6, "Rank by FPS\n(reasoning-aware)", ha="center", fontsize=12, fontweight="bold")

color_for = {r["model_name"]: PROVIDER_COLOR.get(r.get("provider", ""), "#888") for r in results_for_rank}

for i, name in enumerate(aas_order):
    ax.text(0.5, i, f"{i+1}. {short(name)}", ha="center", va="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color_for[name], alpha=0.45, edgecolor="black"))
for i, name in enumerate(fps_order):
    ax.text(2.5, i, f"{i+1}. {short(name)}", ha="center", va="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color_for[name], alpha=0.45, edgecolor="black"))

# arrows
for i, name in enumerate(aas_order):
    j = fps_order.index(name)
    ax.annotate("",
                xy=(2.18, j), xytext=(0.82, i),
                arrowprops=dict(arrowstyle="->", color="grey", alpha=0.55, lw=1.2,
                                connectionstyle="arc3,rad=0.0"))
ax.set_title("Rank flips: 6 of 21 pairwise comparisons reverse direction\n"
             "when reasoning quality enters the score",
             fontsize=12, pad=24, y=1.02)
plt.tight_layout()
plt.savefig(OUT / "rank_flip.png", dpi=160, bbox_inches="tight")
plt.close()

print(f"wrote: {sorted(p.name for p in OUT.glob('*.png'))}")

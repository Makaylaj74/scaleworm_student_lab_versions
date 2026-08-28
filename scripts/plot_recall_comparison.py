"""Recall-recovery figure: how mushroom.pt detects worms on 2023 vs post-2023 footage.

Two panels, both from the two hand-counted validation months (Jan 2023, Feb 2024):
  (A) mean worms you counted vs mean the model found, per month, recall % labelled.
  (B) per-scene scatter vs the y=x "perfect detector" line — 2023 points hug the line,
      post-2023 points collapse to the floor.
Model counts are read from the timeseries cache (mushroom.pt @conf0.25) so this figure
is consistent with figure_worms_over_time_2023_2024.png.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "notebooks/timeseries_results_singleframe"
OUT_PNG = REPO / "notebooks/figure_recall_2023_vs_post2023.png"

BLUE, VERM = "#0072B2", "#D55E00"  # Okabe-Ito: 2023 vs post-2023
MONTHS = [
    ("Jan 2023", "2023 footage", REPO / "notebooks/handcount_2023_01.csv", BLUE),
    (
        "Feb 2024",
        "post-Sep-2023 footage",
        REPO / "notebooks/model_comparison_handcount.csv",
        VERM,
    ),
]


def model_count(stem: str) -> int | None:
    f = CACHE / f"{stem}.json"
    if not f.exists():
        return None
    return json.loads(f.read_text()).get("count")


data = []  # (label, sublabel, color, human[], model[])
for label, sub, csvp, color in MONTHS:
    human, model = [], []
    for r in csv.DictReader(csvp.open()):
        v = str(r.get("human_count", "")).strip()
        mc = model_count(r["stem"])
        if v and mc is not None:
            human.append(int(v))
            model.append(mc)
    data.append((label, sub, color, np.array(human), np.array(model)))
    recall = 100 * sum(model) / sum(human)
    print(
        f"{label}: n={len(human)}  human_mean={np.mean(human):.2f}  model_mean={np.mean(model):.2f}  recall={recall:.0f}%"
    )

fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 5))

# ── Panel A: grouped bars, human vs model mean per month ──
x = np.arange(len(data))
w = 0.36
for i, (label, sub, color, h, m) in enumerate(data):
    axA.bar(
        i - w / 2,
        h.mean(),
        w,
        color="#999999",
        label="you counted (truth)" if i == 0 else None,
    )
    axA.bar(
        i + w / 2,
        m.mean(),
        w,
        color=color,
        label="mushroom.pt found" if i == 0 else None,
    )
    recall = 100 * m.sum() / h.sum()
    axA.text(
        i + w / 2,
        m.mean() + 0.4,
        f"{recall:.0f}%\nrecall",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=color,
    )
    axA.text(
        i - w / 2,
        h.mean() + 0.4,
        f"{h.mean():.1f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )
axA.set_xticks(x)
axA.set_xticklabels([f"{lab}\n({sub})" for lab, sub, *_ in data], fontsize=10)
axA.set_ylabel("Mean scale worms per scene", fontweight="bold", fontsize=12)
axA.set_title(
    "A. Worms counted vs detected", fontsize=12, fontweight="bold", loc="left"
)
axA.set_ylim(0, 18)
axA.legend(framealpha=0.9, loc="upper right", fontsize=10)
axA.grid(axis="y", alpha=0.3)

# ── Panel B: per-scene scatter vs y = x ──
hi = max(max(h.max(), m.max()) for _, _, _, h, m in data) + 2
axB.plot(
    [0, hi],
    [0, hi],
    "--",
    color="#444444",
    lw=1.3,
    zorder=1,
    label="perfect detector (y = x)",
)
for label, sub, color, h, m in data:
    recall = 100 * m.sum() / h.sum()
    axB.scatter(
        h,
        m,
        s=55,
        color=color,
        alpha=0.75,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
        label=f"{label} — {recall:.0f}% recall",
    )
axB.set_xlim(0, hi)
axB.set_ylim(0, hi)
axB.set_aspect("equal")
axB.set_xlabel("You counted (worms/scene)", fontweight="bold", fontsize=12)
axB.set_ylabel("mushroom.pt found (worms/scene)", fontweight="bold", fontsize=12)
axB.set_title(
    "B. Per-scene: on the line = recovered, on the floor = missed",
    fontsize=12,
    fontweight="bold",
    loc="left",
)
axB.legend(framealpha=0.9, loc="upper left", fontsize=10)
axB.grid(alpha=0.3)

fig.suptitle(
    "mushroom.pt scale-worm recall: 2023 vs post-Sep-2023 footage",
    fontsize=13,
    fontweight="bold",
)
for ax in (axA, axB):
    for s in ax.spines.values():
        s.set_linewidth(1.0)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
print(f"wrote {OUT_PNG}")

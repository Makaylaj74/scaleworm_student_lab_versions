"""Inter-rater reliability figure (Round 2, count/box): Makayla vs Rater B.

INTERIM + CONFOUNDED: MJ's reference counts are box_corrected from detector pre-labels; Rater B (LG)
counted from scratch, so part of the gap is method, not rater. The clean comparison needs MJ's own
blind from-scratch packet (irr_packet_MJ, not yet done). Labelled as such on the figure.

Two panels: (A) per-frame count scatter with 1:1 line (ICC(2,1), Pearson r, bias annotated);
(B) Bland-Altman agreement (mean vs difference, bias line + 95% limits of agreement).

Output: notebooks/figure_irr_mj_vs_lg.png (300 dpi).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/home/jovyan/scaleworm-student-lab")
MJ = REPO / "validation/irr_round_2/raters/MJ_monday"
LG = REPO / "notebooks/LG_results"
C_PT, C_LINE = "#0072B2", "#D55E00"


def counts(d: Path) -> dict[str, int]:
    return {lp.stem: sum(1 for ln in lp.read_text().splitlines() if ln.strip())
            for lp in (d / "labels").glob("*.txt")}


def boxes(lp: Path):
    out = []
    for ln in lp.read_text().splitlines():
        if ln.strip():
            _, cx, cy, w, h = ln.split()
            out.append((float(cx), float(cy), float(w), float(h)))
    return out


def box_f1() -> tuple[float, float, float]:
    """Pooled centre-in-box match, LG boxes vs MJ boxes (normalized coords)."""
    tp = fp = fn = 0
    for stem in set(counts(MJ)) & set(counts(LG)):
        mj, lg = boxes(MJ / "labels" / f"{stem}.txt"), boxes(LG / "labels" / f"{stem}.txt")
        claimed = [False] * len(mj)
        hit = 0
        for cx, cy, _w, _h in lg:
            for j, (mx, my, mw, mh) in enumerate(mj):
                if not claimed[j] and abs(cx - mx) <= mw / 2 and abs(cy - my) <= mh / 2:
                    claimed[j] = True
                    hit += 1
                    break
        tp += hit
        fp += len(lg) - hit
        fn += len(mj) - hit
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    return p, r, 2 * p * r / (p + r) if p + r else 0


def icc21(x, y) -> float:
    """ICC(2,1): two-way random, single measures, absolute agreement."""
    m = np.column_stack([x, y]).astype(float)
    n, k = m.shape
    gm = m.mean()
    msr = k * ((m.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((m.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((m - m.mean(1, keepdims=True) - m.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def main() -> None:
    cm, cl = counts(MJ), counts(LG)
    stems = sorted(set(cm) & set(cl))
    x = np.array([cm[s] for s in stems], float)  # MJ
    y = np.array([cl[s] for s in stems], float)  # LG
    r = np.corrcoef(x, y)[0, 1]
    bias = (y - x).mean()
    icc = icc21(x, y)
    p, rec, f1 = box_f1()

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 5.2))

    lim = max(x.max(), y.max()) * 1.1
    axA.plot([0, lim], [0, lim], color="0.6", ls="--", lw=1, label="1:1")
    axA.scatter(x, y, s=40, color=C_PT, alpha=0.7, edgecolor="white", linewidth=0.5)
    axA.set_xlim(0, lim)
    axA.set_ylim(0, lim)
    axA.set_aspect("equal")
    axA.set_xlabel("Makayla count (worms / frame)", fontweight="bold", fontsize=11)
    axA.set_ylabel("Rater B count (worms / frame)", fontweight="bold", fontsize=11)
    axA.set_title("A  Per-frame counts", fontsize=11, loc="left", fontweight="bold")
    axA.grid(alpha=0.3)
    axA.text(0.04, 0.96, f"n = {len(stems)}\nICC(2,1) = {icc:.2f}\nr = {r:.2f}\n"
             f"bias = {bias:+.1f}/fr\nbox F1 = {f1:.2f}",
             transform=axA.transAxes, va="top", ha="left", fontsize=9,
             bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"})
    axA.legend(loc="lower right", framealpha=0.9, fontsize=9)

    mean_xy = (x + y) / 2
    diff = y - x  # LG - MJ
    md, sd = diff.mean(), diff.std(ddof=1)
    axB.axhline(0, color="0.6", lw=1)
    axB.axhline(md, color=C_LINE, lw=1.5, label=f"mean diff {md:+.1f}")
    axB.axhline(md + 1.96 * sd, color=C_LINE, ls="--", lw=1, label="95% limits of agreement")
    axB.axhline(md - 1.96 * sd, color=C_LINE, ls="--", lw=1)
    axB.scatter(mean_xy, diff, s=40, color=C_PT, alpha=0.7, edgecolor="white", linewidth=0.5)
    axB.set_xlabel("Mean of the two counts (worms / frame)", fontweight="bold", fontsize=11)
    axB.set_ylabel("Rater B − Makayla (worms / frame)", fontweight="bold", fontsize=11)
    axB.set_title("B  Bland–Altman agreement", fontsize=11, loc="left", fontweight="bold")
    axB.grid(alpha=0.3)
    axB.legend(loc="lower left", framealpha=0.9, fontsize=8)

    for ax in (axA, axB):
        for sp in ax.spines.values():
            sp.set_linewidth(1.0)

    fig.suptitle("Inter-rater reliability (count/box), Round 2 — Makayla vs Rater B  "
                 "[INTERIM, method-confounded]", fontsize=12.5, fontweight="bold")
    cap = ("AI-generated caption (Claude, Anthropic) — Round-2 count/box inter-rater reliability on "
           f"{len(stems)} jointly-counted frames (low Jan-2022 vs peak Jan-2023), scale-worm counts, "
           "CAMHDA301. (A) Per-frame counts, Makayla vs Rater B, with the 1:1 line; ICC(2,1) absolute "
           "agreement, Pearson r, mean bias, and pooled centre-in-box F1 annotated. (B) Bland-Altman: "
           "difference (Rater B − Makayla) vs the pair mean, with mean bias and 95% limits of agreement. "
           "INTERIM AND CONFOUNDED: Makayla's reference labels are box-corrected from detector pre-labels "
           "while Rater B counted from scratch, so part of the systematic under-count is method, not rater. The "
           "clean estimate requires Makayla's own blind from-scratch packet (irr_packet_MJ, pending). "
           "Sources: validation/irr_round_2/raters/MJ_monday, notebooks/LG_results.")
    fig.subplots_adjust(bottom=0.34, top=0.86, wspace=0.28)
    fig.text(0.5, 0.015, cap, ha="center", va="bottom", wrap=True, fontsize=7.5)
    fig.savefig(REPO / "notebooks/figure_irr_mj_vs_lg.png", dpi=300, bbox_inches="tight")
    print(f"wrote figure_irr_mj_vs_lg.png  n={len(stems)} ICC={icc:.2f} r={r:.2f} "
          f"bias={bias:+.1f} boxF1={f1:.2f} (P={p:.2f} R={rec:.2f})")


if __name__ == "__main__":
    main()

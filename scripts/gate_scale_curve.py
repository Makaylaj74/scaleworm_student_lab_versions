"""Tier-B learning curve: does dense-blurry recall scale with more labels, or is it a ceiling?

The three nested models from build_scale_datasets.py (clear + first N Tuesday-blurry labels,
N in {15, 30, 47}) were trained but never scored. This gates all three on the SAME held-out 23
dense Tuesday frames used by gate_v6_blurry (datasets/scaleworm_v6_blurry/tuesday_blurry_test.txt),
with the identical point-matched metric + bootstrap CIs, and plots blurry recall vs N.

Decision this answers:
  - recall still CLIMBING at N=47  -> dense IS label-fixable -> worth labeling the remaining 59
  - recall FLAT across N           -> density ceiling confirmed -> blurry/dense eras need a
                                      different approach (restoration / %-cover target / manual)

v5_all (the deployed model, trained without these dense Tuesday labels) is scored as the
baseline reference line. Reuses gate_v6_blurry's tested predict/match/report helpers verbatim.

Usage (thesis venv, ultralytics 8.4.62, matches all baselines):
  /home/jovyan/joseph-scaleworm-thesis/.venv/bin/python3 scripts/gate_scale_curve.py
"""

from __future__ import annotations

import csv
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gate_retrain import day_of
from gate_v6_blurry import TEST_LIST, blurry_recs, report

REPO = Path("/home/jovyan/scaleworm-student-lab")
OUT_CSV = REPO / "validation" / "scale_curve_gate.csv"
OUT_FIG = REPO / "notebooks" / "figure_scale_learning_curve.png"

SIZES = [15, 30, 47]
SCALE_MODELS = {
    n: REPO / "99_runs" / f"scaleworm_v6_scale_n{n}" / "weights" / "best.pt"
    for n in SIZES
}
V5_ALL = REPO / "99_runs" / "scaleworm_v5_all" / "weights" / "best.pt"

DET_C, REF_C, CNT_C = "#0072B2", "#999999", "#D55E00"


def training_days(n: int) -> set[str]:
    ds = REPO / "datasets" / f"scaleworm_v6_scale_n{n}"
    days = set()
    for split in ("train", "val"):
        for p in glob.glob(str(ds / "labels" / split / "*.txt")):
            days.add(day_of(Path(p).stem))
    return days


def verify_no_leakage(test: list[str]) -> None:
    for n in SIZES:
        tdays = training_days(n)
        leaked = [g for g in test if day_of(g) in tdays]
        assert not leaked, f"LEAKAGE in n{n}: {leaked}"
    print(
        f"leakage OK: 0/{len(test)} held-out Tuesday frames share a day with any scale-model train set"
    )


def main() -> None:
    from ultralytics import YOLO

    test = [s.strip() for s in TEST_LIST.read_text().splitlines() if s.strip()]
    verify_no_leakage(test)

    rows = []
    # baseline: v5_all (deployed, no dense Tuesday labels)
    base = report(
        "v5_all (baseline)",
        "BLURRY (Tuesday-dense, MJ-box GT)",
        blurry_recs(YOLO(str(V5_ALL))),
    )
    base["n_blurry_labels"] = 0
    rows.append(base)
    # learning curve
    for n in SIZES:
        r = report(
            f"v6_scale_n{n}",
            "BLURRY (Tuesday-dense, MJ-box GT)",
            blurry_recs(YOLO(str(SCALE_MODELS[n]))),
        )
        r["n_blurry_labels"] = n
        rows.append(r)

    with OUT_CSV.open("w", newline="") as f:
        fields = [
            "n_blurry_labels",
            "model",
            "era",
            "conf",
            "n",
            "human",
            "model_total",
            "count_recall",
            "point_recall",
            "point_recall_ci_lo",
            "point_recall_ci_hi",
            "precision",
            "f1",
            "mae",
            "bias",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT_CSV}")

    # --- learning-curve figure ---
    curve = [r for r in rows if r["n_blurry_labels"] > 0]
    xs = [r["n_blurry_labels"] for r in curve]
    pr = [r["point_recall"] for r in curve]
    lo = [r["point_recall"] - r["point_recall_ci_lo"] for r in curve]
    hi = [r["point_recall_ci_hi"] - r["point_recall"] for r in curve]
    cr = [r["count_recall"] for r in curve]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.errorbar(
        xs,
        pr,
        yerr=[lo, hi],
        fmt="o-",
        color=DET_C,
        lw=2.2,
        ms=8,
        capsize=5,
        label="point recall (95% CI)",
    )
    ax.plot(
        xs, cr, "s--", color=CNT_C, lw=1.6, ms=6, alpha=0.8, label="count-ratio recall"
    )
    ax.axhline(
        base["point_recall"],
        color=REF_C,
        lw=1.5,
        ls=":",
        label=f"v5_all baseline (point {base['point_recall']:.0%})",
    )
    for x, y in zip(xs, pr):
        ax.annotate(
            f"{y:.0%}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            color=DET_C,
            fontweight="bold",
        )

    ax.set_xlabel(
        "Dense Tuesday-blurry training frames (N)", fontweight="bold", fontsize=12
    )
    ax.set_ylabel(
        "Blurry recall on held-out 23 Tuesday frames", fontweight="bold", fontsize=12
    )
    ax.set_title(
        "Tier-B learning curve: does more dense-blurry labeling lift recall?\n"
        "Held-out dense Tuesday (conf 0.25, point-matched vs MJ box centres)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xticks(xs)
    ax.set_ylim(0, max(max(pr), max(cr), base["point_recall"]) * 1.25)
    ax.grid(alpha=0.3)
    ax.legend(framealpha=0.9, fontsize=10, loc="upper left")
    for s in ax.spines.values():
        s.set_linewidth(1.0)

    slope = pr[-1] - pr[0]
    flat = (
        abs(slope) < 0.03
        and curve[0]["point_recall_ci_lo"] <= pr[-1] <= curve[0]["point_recall_ci_hi"]
    )
    verdict = (
        "DENSITY CEILING: recall flat across N"
        if flat
        else f"recall {'rises' if slope > 0 else 'falls'} {slope:+.0%} N15->N47"
    )
    ax.text(
        0.98,
        0.02,
        verdict,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        style="italic",
        color="0.3",
        bbox={"boxstyle": "round", "fc": "white", "ec": "0.7", "alpha": 0.9},
    )

    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    print(f"wrote {OUT_FIG}")
    print(f"\nLEARNING-CURVE VERDICT: {verdict}")
    print(
        f"  n15={pr[0]:.1%}  n30={pr[1]:.1%}  n47={pr[2]:.1%}  (slope {slope:+.1%})  "
        f"| v5_all baseline {base['point_recall']:.1%}"
    )


if __name__ == "__main__":
    main()

"""Figure: detector recall by camera unit / regime (the gate result), for Dax.

Panel A: model-vs-human scatter for the clear-window frames, coloured by camera unit
(-2022 vs -2021), y=x.
Panel B: recall (model total / human total) with bootstrap 95% CI for four hand-count
sets — the two clear units (-2022, -2021) and the two blurry post-2023 months — showing
recall is unit-dependent: strong on -2022, moderate on -2021, near-zero when blurry.

All sets scored identically here (mushroom.pt, conf=0.25) so recalls are comparable.
Run under the thesis venv (ultralytics 8.4.62):
    /home/jovyan/joseph-scaleworm-thesis/.venv/bin/python scripts/plot_recall_gate_comparison.py
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from ultralytics import YOLO

REPO = Path("/home/jovyan/scaleworm-student-lab")
sys.path.insert(0, str(REPO / "scripts"))
from count_frames import count_worms

CONF = 0.25
BOOT_N = 10_000
SEED = 20260908
OK = {
    "blue": "#0072B2",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "orange": "#E69F00",
    "grey": "#999999",
}

CW_FRAMES = REPO / "validation/clear_window_handcount/frames"
CW_SHEET = REPO / "validation/clear_window_handcount/handcount_sheet.csv"

# (label, colour, frames_dir, human_csv, id_col, human_col, unit_filter)
SETS = [
    ("−2022 clear\n(2023)", OK["green"], CW_FRAMES, CW_SHEET, "frame_id", "worm_count", "CAMHDA301-2022"),
    ("−2021 clear\n(2021–22)", OK["blue"], CW_FRAMES, CW_SHEET, "frame_id", "worm_count", "CAMHDA301-2021"),
    ("Feb 2024\nblurry", OK["vermillion"], REPO / "notebooks/model_comparison_frames",
     REPO / "notebooks/model_comparison_handcount.csv", "stem", "human_count", None),
    ("May 2024\nblurry", OK["vermillion"], REPO / "notebooks/handcount_2024_05_frames",
     REPO / "notebooks/handcount_2024_05.csv", "stem", "human_count", None),
]


def load_pairs(model, frames_dir, human_csv, id_col, human_col, unit=None):
    pairs = []
    for r in csv.DictReader(human_csv.open()):
        if unit is not None and r.get("camera_unit") != unit:
            continue
        h = r[human_col].strip()
        if not h:
            continue
        png = frames_dir / f"{r[id_col]}.png"
        if not png.exists():
            continue
        res = model(str(png), conf=CONF, verbose=False)[0]
        cls_ids = [int(c) for c in res.boxes.cls.tolist()]
        pairs.append((count_worms(cls_ids, model.names), int(h)))
    return pairs


def recall(pairs):
    ht = sum(h for _, h in pairs)
    return sum(m for m, _ in pairs) / ht if ht else 0.0


def boot_ci(pairs, n=BOOT_N, seed=SEED):
    rng = random.Random(seed)
    k = len(pairs)
    vals = sorted(recall([pairs[rng.randrange(k)] for _ in range(k)]) for _ in range(n))
    return vals[int(0.025 * n)], vals[int(0.975 * n)]


def main() -> None:
    model = YOLO(str(REPO / "mushroom.pt"))
    data = []
    for label, colour, fdir, hcsv, idc, hc, unit in SETS:
        pairs = load_pairs(model, fdir, hcsv, idc, hc, unit)
        lo, hi = boot_ci(pairs)
        data.append(
            {"label": label, "colour": colour, "pairs": pairs,
             "recall": recall(pairs), "lo": lo, "hi": hi, "n": len(pairs)}
        )
        print(f"{label.splitlines()[0]:16s} n={len(pairs):3d} recall={recall(pairs):.1%} CI[{lo:.1%},{hi:.1%}]")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.6))

    # Panel A — clear-window scatter, coloured by unit
    u22, u21 = data[0]["pairs"], data[1]["pairs"]
    hi = max(max(h for _, h in u22 + u21), max(m for m, _ in u22 + u21)) + 2
    axA.plot([0, hi], [0, hi], ls="--", c=OK["grey"], lw=1, label="1:1 (perfect)")
    axA.scatter([h for _, h in u22], [m for m, _ in u22], s=55, c=OK["green"],
                edgecolor="k", linewidth=0.4, alpha=0.85, zorder=3, label=f"−2022 ({data[0]['recall']:.0%})")
    axA.scatter([h for _, h in u21], [m for m, _ in u21], s=55, c=OK["blue"],
                edgecolor="k", linewidth=0.4, alpha=0.85, zorder=3, label=f"−2021 ({data[1]['recall']:.0%})")
    axA.set_xlim(0, hi)
    axA.set_ylim(0, hi)
    axA.set_aspect("equal")
    axA.set_xlabel("Human count (worms / frame)")
    axA.set_ylabel("mushroom.pt count (worms / frame)")
    axA.set_title("A. Clear-window frames by camera unit\nboth under-count; −2021 more so", fontsize=10)
    axA.legend(fontsize=8, loc="upper left")
    axA.grid(True, alpha=0.25)

    # Panel B — recall bars
    labels = [d["label"] for d in data]
    recalls = [d["recall"] * 100 for d in data]
    err_lo = [(d["recall"] - d["lo"]) * 100 for d in data]
    err_hi = [(d["hi"] - d["recall"]) * 100 for d in data]
    colours = [d["colour"] for d in data]
    x = range(len(data))
    axB.bar(x, recalls, yerr=[err_lo, err_hi], color=colours, edgecolor="k",
            linewidth=0.5, capsize=4, width=0.68)
    for i, d in enumerate(data):
        axB.text(i, d["recall"] * 100 + err_hi[i] + 2.5, f"{d['recall']:.0%}",
                 ha="center", fontsize=9, fontweight="bold")
    axB.set_xticks(list(x))
    axB.set_xticklabels(labels, fontsize=8)
    axB.set_ylabel("Detector recall  (model total / human total)")
    axB.set_ylim(0, 100)
    axB.set_title("B. Recall by unit / regime\n(conf=0.25, error bars = bootstrap 95% CI)", fontsize=10)
    axB.legend(
        handles=[
            Patch(facecolor=OK["green"], edgecolor="k", label="−2022 clear"),
            Patch(facecolor=OK["blue"], edgecolor="k", label="−2021 clear"),
            Patch(facecolor=OK["vermillion"], edgecolor="k", label="blurry (post-2023)"),
        ],
        fontsize=8, loc="upper right",
    )
    axB.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    out = REPO / "notebooks/figure_recall_gate_clear_vs_blurry.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("wrote", out)

    with (REPO / "validation/clear_window_handcount/recall_gate_comparison.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set", "n_frames", "human_total", "model_total", "recall", "ci_lo", "ci_hi"])
        for d in data:
            w.writerow([d["label"].replace("\n", " "), d["n"],
                        sum(h for _, h in d["pairs"]), sum(m for m, _ in d["pairs"]),
                        f"{d['recall']:.4f}", f"{d['lo']:.4f}", f"{d['hi']:.4f}"])
    print(f"# conf={CONF} ultralytics=8.4.62 seed={SEED} boot={BOOT_N}")


if __name__ == "__main__":
    main()

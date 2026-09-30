"""Spot-check overlay: are v5_all's 'extra' clear boxes real worms or phantoms?

For the clear gate frames with the most unmatched (FP-vs-clicks) boxes, overlays on the
frame: MJ clicks (yellow x), model boxes that DID contain a click (green = TP), model
boxes with NO click inside (red = 'FP' vs clicks), and clicks with no box (blue o = FN).
If the red boxes sit on visible worms MJ simply did not click, the count-ratio 'over-count'
is really the model recovering worms the human missed, not phantom detections.

Usage (thesis venv):
  CONF=0.25 MODEL=99_runs/scaleworm_v5_all/weights/best.pt \
    /home/jovyan/joseph-scaleworm-thesis/.venv/bin/python3 scripts/spotcheck_retrain.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

REPO = Path("/home/jovyan/scaleworm-student-lab")
sys.path.insert(0, str(REPO / "scripts"))
from gate_retrain import CLEAR, point_match

MODEL = os.environ.get("MODEL", str(REPO / "99_runs/scaleworm_v5_all/weights/best.pt"))
CONF = float(os.environ.get("CONF", "0.25"))
N_SHOW = int(os.environ.get("N_SHOW", "6"))
OUT = Path(
    os.environ.get(
        "OUT", str(REPO / "validation" / "retrain_spotcheck_v5_all_clear.png")
    )
)

# Okabe-Ito
GREEN = "#009E73"  # TP box (has a click)
RED = "#D55E00"  # unmatched box (FP vs clicks)
YELLOW = "#F0E442"  # click
BLUE = "#0072B2"  # FN click (no box)


def matched_flags(preds, points):
    """Return (box_is_tp, click_is_matched) mirroring gate_retrain.point_match order."""
    claimed = [False] * len(points)
    box_tp = []
    for x1, y1, x2, y2, _c in sorted(preds, key=lambda p: -p[4]):
        hit = False
        for j, (px, py) in enumerate(points):
            if not claimed[j] and x1 <= px <= x2 and y1 <= py <= y2:
                claimed[j] = True
                hit = True
                break
        box_tp.append((x1, y1, x2, y2, hit))
    return box_tp, claimed


def main() -> None:
    from ultralytics import YOLO

    rows = [
        r
        for r in csv.DictReader(CLEAR.joinpath("handcount_sheet.csv").open())
        if r.get("frame_ready") == "Y" and (r.get("worm_count") or "").strip() != ""
    ]
    model = YOLO(MODEL)
    frames = []
    for r in rows:
        fid = r["frame_id"]
        png = CLEAR / "frames" / f"{fid}.png"
        clk = CLEAR / "clicks" / f"{fid}.json"
        if not png.exists() or not clk.exists():
            continue
        pts = [tuple(p) for p in json.load(clk.open())["clicks"]]
        res = model(str(png), conf=CONF, verbose=False)[0]
        preds = [
            (float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c))
            for b, c in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist())
        ]
        _tp, fp, _fn = point_match(preds, pts)
        frames.append({"fid": fid, "png": png, "pts": pts, "preds": preds, "fp": fp})

    frames.sort(key=lambda d: -d["fp"])
    show = frames[:N_SHOW]
    total_fp = sum(f["fp"] for f in frames)
    print(
        f"clear frames={len(frames)}  total unmatched(FP-vs-clicks)={total_fp} @ conf {CONF}"
    )
    print("showing highest-FP frames:", [(f["fid"], f["fp"]) for f in show])

    ncol = 3
    nrow = (len(show) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 4.2 * nrow))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for ax, f in zip(axes, show):
        img = plt.imread(str(f["png"]))
        ax.imshow(img)
        box_tp, click_ok = matched_flags(f["preds"], f["pts"])
        for x1, y1, x2, y2, hit in box_tp:
            ax.add_patch(
                mpatches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor=GREEN if hit else RED,
                    linewidth=1.6,
                )
            )
        for (px, py), ok in zip(f["pts"], click_ok):
            ax.plot(
                px,
                py,
                "x" if ok else "o",
                color=YELLOW if ok else BLUE,
                markersize=7,
                markeredgewidth=2,
            )
        nfp = sum(1 for *_, hit in box_tp if not hit)
        ax.set_title(
            f"{f['fid'][9:22]}  clicks={len(f['pts'])} boxes={len(f['preds'])} unmatched={nfp}",
            fontsize=9,
        )
        ax.axis("off")
    for ax in axes[len(show) :]:
        ax.axis("off")
    legend = [
        mpatches.Patch(color=GREEN, label="box w/ click (TP)"),
        mpatches.Patch(color=RED, label="box, NO click (FP vs clicks)"),
        mpatches.Patch(color=YELLOW, label="click matched"),
        mpatches.Patch(color=BLUE, label="click, no box (FN)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=9)
    fig.suptitle(
        f"v5_all clear spot-check @ conf {CONF} — are red boxes real worms MJ didn't click?",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

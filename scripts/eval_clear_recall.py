"""v2 detector recall on clean CLEAR-window frames with INDEPENDENT click ground truth.

Companion to ``eval_val_recall.py`` (which measured v3 on blurry Monday-val frames).
This scores v2 on the clear-window hand-count set (``validation/clear_window_handcount``):
2021-2023 clear Scene-1 frames the counter marked by eye (click points), NOT by
correcting pre-labels -- so unlike the Monday-val check, BOTH recall and precision are
independent here.

Leakage control is DAY-LEVEL (same rule as eval_val_recall): drop any hand-count frame
whose day contributed a frame to the v2/v3 training set. All clear frames precede the
2023-08-10 blur onset, so all route to v2.

Ground truth is POINTS (clicks), predictions are BOXES: a prediction hits a worm when a
click falls inside the predicted box (greedy by descending confidence, each click
claimed once). Reports recall / precision / F1 + count MAE / bias, bootstrap 95% CI
(percentile, n=10000, seed=20260913).

Run on the GPU box (student-lab venv, L40S).
"""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import date
from pathlib import Path

REPO = Path("/home/jovyan/scaleworm-student-lab")
sys.path.insert(0, str(REPO / "scripts"))
from eval_val_recall import _boot_ci, training_days

CONF = float(os.environ.get("CONF", "0.25"))
BLUR_ONSET = date(2023, 8, 10)
BASE = REPO / "validation/clear_window_handcount"
SHEET = BASE / "handcount_sheet.csv"
CLICKS = BASE / "clicks"
FRAMES = BASE / "frames"
V2 = REPO / "99_runs/scaleworm_v2/weights/best.pt"


def point_match(preds, points) -> tuple[int, int, int]:
    """Greedy click-in-box match. preds: (x1,y1,x2,y2,conf); points: (px,py).

    Returns (tp, fp, fn). A prediction hits a click if the click lies inside the
    prediction box; each click is claimed once, predictions processed by conf desc.
    """
    claimed = [False] * len(points)
    tp = 0
    for x1, y1, x2, y2, _conf in sorted(preds, key=lambda p: -p[4]):
        for j, (px, py) in enumerate(points):
            if not claimed[j] and x1 <= px <= x2 and y1 <= py <= y2:
                claimed[j] = True
                tp += 1
                break
    fp = len(preds) - tp
    fn = len(points) - tp
    return tp, fp, fn


def main() -> None:
    import numpy as np
    from ultralytics import YOLO

    rng = np.random.default_rng(20260913)
    tdays = training_days()
    rows = [
        r
        for r in csv.DictReader(SHEET.open(newline=""))
        if (r.get("worm_count") or "").strip() not in ("", "NA")
    ]

    def day(fid: str) -> str:
        return fid.split("-")[1][:8]

    clean = [r for r in rows if day(r["frame_id"]) not in tdays]
    print(f"counted clear frames={len(rows)}  clean(day-unseen)={len(clean)}  "
          f"excluded={len(rows) - len(clean)}")

    model = YOLO(str(V2))
    recs = []
    for r in clean:
        fid = r["frame_id"]
        png = FRAMES / f"{fid}.png"
        clk = CLICKS / f"{fid}.json"
        if not png.exists() or not clk.exists():
            print(f"  skip {fid}: missing frame or clicks")
            continue
        points = [tuple(p) for p in json.load(clk.open())["clicks"]]
        res = model(str(png), conf=CONF, verbose=False)[0]
        preds = [
            (float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c))
            for b, c in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist())
        ]
        tp, fp, fn = point_match(preds, points)
        recs.append(
            {"fid": fid, "gt": len(points), "pred": len(preds), "tp": tp, "fp": fp, "fn": fn}
        )

    n = len(recs)
    TP = sum(x["tp"] for x in recs)
    FP = sum(x["fp"] for x in recs)
    FN = sum(x["fn"] for x in recs)
    gt = np.array([x["gt"] for x in recs], float)
    pred = np.array([x["pred"] for x in recs], float)
    recall = TP / (TP + FN)
    prec = TP / (TP + FP) if TP + FP else float("nan")
    f1 = 2 * prec * recall / (prec + recall)
    fr_recall = np.array([x["tp"] / x["gt"] for x in recs if x["gt"]])
    rec_ci = _boot_ci(fr_recall, np.mean, rng)
    mae_ci = _boot_ci(np.abs(pred - gt), np.mean, rng)
    prec_frame = np.array([x["tp"] / x["pred"] for x in recs if x["pred"]])
    prec_ci = _boot_ci(prec_frame, np.mean, rng)

    print(f"\n[v2 CLEAR] clean frames n={n}  (independent-click worms={int(gt.sum())})")
    print(f"  detection recall = {recall:.2%}  (TP={TP} FN={FN})  "
          f"per-frame 95%CI [{rec_ci[0]:.2%}, {rec_ci[1]:.2%}]")
    print(f"  precision        = {prec:.2%}  (FP={FP})  per-frame 95%CI "
          f"[{prec_ci[0]:.2%}, {prec_ci[1]:.2%}]   F1={f1:.2%}")
    print(f"  count MAE        = {np.mean(np.abs(pred - gt)):.2f} worms/frame  95%CI "
          f"[{mae_ci[0]:.2f}, {mae_ci[1]:.2f}]")
    print(f"  count bias       = {np.mean(pred - gt):+.2f} (pred - human)   "
          f"model total {int(pred.sum())} vs human {int(gt.sum())} "
          f"({pred.sum() / gt.sum():.1%})")


if __name__ == "__main__":
    main()

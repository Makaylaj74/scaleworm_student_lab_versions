"""Confidence-threshold sweep for a retrained detector on the held-out gate.

Runs inference ONCE per gate frame at a very low conf floor, caches (boxes, confs) +
ground-truth points, then sweeps operating thresholds in post — so the whole sweep costs
one forward pass per frame. Reports, per era and per threshold, point-matched
recall/precision/F1 and the count-ratio (model/human). Picks the threshold whose
count-ratio is closest to 100% (best for an unbiased abundance index) and the one that
maximizes F1.

CLEAR ground truth = independent clicks; BLURRY ground truth = held-out MJ box centers.

Usage (thesis venv):
  MODEL=99_runs/scaleworm_v5_all/weights/best.pt LABEL=v5_all \
    /home/jovyan/joseph-scaleworm-thesis/.venv/bin/python3 scripts/sweep_conf_retrain.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

REPO = Path("/home/jovyan/scaleworm-student-lab")
sys.path.insert(0, str(REPO / "scripts"))
from gate_retrain import (
    BLURRY_SHEET,
    CLEAR,
    MONDAY,
    point_match,
    yolo_box_centers,
)

MODEL = os.environ.get("MODEL", str(REPO / "mushroom.pt"))
LABEL = os.environ.get("LABEL", Path(MODEL).stem)
FLOOR = 0.01
THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]


def collect(model, era: str):
    """Return list of (preds_all, gt_points) per frame for an era, at the conf floor."""
    out = []
    if era == "clear":
        rows = [
            r
            for r in csv.DictReader(CLEAR.joinpath("handcount_sheet.csv").open())
            if r.get("frame_ready") == "Y" and (r.get("worm_count") or "").strip() != ""
        ]
        for r in rows:
            fid = r["frame_id"]
            png = CLEAR / "frames" / f"{fid}.png"
            clk = CLEAR / "clicks" / f"{fid}.json"
            if not png.exists() or not clk.exists():
                continue
            pts = [tuple(p) for p in json.load(clk.open())["clicks"]]
            res = model(str(png), conf=FLOOR, verbose=False)[0]
            preds = [
                (float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c))
                for b, c in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist())
            ]
            out.append((preds, pts))
    else:
        rows = [
            r
            for r in csv.DictReader(BLURRY_SHEET.open())
            if (r.get("human_count") or "").strip() not in ("", "nan")
        ]
        for r in rows:
            fid = r["stem"]
            png = MONDAY / "images" / f"{fid}.png"
            if not png.exists() or not (MONDAY / "labels" / f"{fid}.txt").exists():
                continue
            res = model(str(png), conf=FLOOR, verbose=False)[0]
            H, W = res.orig_shape
            pts = yolo_box_centers(fid, W, H)
            preds = [
                (float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c))
                for b, c in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist())
            ]
            out.append((preds, pts))
    return out


def sweep(frames):
    rows = []
    human_tot = sum(len(pts) for _, pts in frames)
    for t in THRESHOLDS:
        TP = FP = FN = model_tot = 0
        for preds_all, pts in frames:
            preds = [p for p in preds_all if p[4] >= t]
            tp, fp, fn = point_match(preds, pts)
            TP += tp
            FP += fp
            FN += fn
            model_tot += len(preds)
        rec = TP / (TP + FN) if TP + FN else 0.0
        prec = TP / (TP + FP) if TP + FP else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        ratio = model_tot / human_tot if human_tot else 0.0
        rows.append((t, rec, prec, f1, ratio, model_tot, human_tot))
    return rows


def main() -> None:
    from ultralytics import YOLO

    model = YOLO(MODEL)
    for era in ("clear", "blurry"):
        frames = collect(model, era)
        rows = sweep(frames)
        print(f"\n=== {LABEL} · {era.upper()}  (n={len(frames)} frames) ===")
        print(
            f"  {'conf':>5} {'recall':>7} {'prec':>7} {'F1':>7} {'count%':>8} {'model/human':>12}"
        )
        for t, rec, prec, f1, ratio, mt, ht in rows:
            print(
                f"  {t:>5.2f} {rec:>7.1%} {prec:>7.1%} {f1:>7.1%} {ratio:>8.0%} {mt:>6}/{ht:<6}"
            )
        best_f1 = max(rows, key=lambda r: r[3])
        best_cal = min(rows, key=lambda r: abs(r[4] - 1.0))
        print(
            f"  -> max-F1  @ conf {best_f1[0]:.2f}: F1 {best_f1[3]:.1%} "
            f"(rec {best_f1[1]:.1%}, prec {best_f1[2]:.1%}, count {best_f1[4]:.0%})"
        )
        print(
            f"  -> unbiased@ conf {best_cal[0]:.2f}: count {best_cal[4]:.0%} "
            f"(rec {best_cal[1]:.1%}, prec {best_cal[2]:.1%}, F1 {best_cal[3]:.1%})"
        )


if __name__ == "__main__":
    main()

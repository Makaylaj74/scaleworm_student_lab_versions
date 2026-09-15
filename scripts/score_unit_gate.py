"""Per-camera-unit detector gate: score v2 AND v3 against INDEPENDENT hand counts
and recommend which model to use for that unit.

Given a gate directory holding, for a single camera-unit:
  <gate>/handcount_sheet.csv   (frame_id, worm_count, ...; independent counts)
  <gate>/clicks/<frame_id>.json ({"clicks": [[px,py], ...]}) -- the ground truth
  <gate>/frames/<frame_id>.png  (the extracted Scene-1 frame)
this runs each model at CONF, matches predictions to the click ground truth with
the same click-in-box greedy rule used in eval_clear_recall.py, and reports per
model: detection recall / precision / F1 (bootstrap 95% CI), plus count bias /
MAE. It then RECOMMENDS the model with the best count agreement among those whose
recall clears MIN_RECALL, and prints the recall-correction factor (1/recall).

Ground truth is INDEPENDENT clicks (counted blind, not by editing model boxes),
so both recall and precision are meaningful -- see the gate protocol note.

Leakage: frames whose day is in the v2/v3 training set are dropped by default
(--keep-training to override).

Env: V2_MODEL, V3_MODEL, CONF (0.25), MIN_RECALL (0.40), SEED (20260915).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from eval_clear_recall import point_match
from eval_val_recall import _boot_ci, training_days

CONF = float(os.environ.get("CONF", "0.25"))
MIN_RECALL = float(os.environ.get("MIN_RECALL", "0.40"))
SEED = int(os.environ.get("SEED", "20260915"))
MODELS = {
    "v2": Path(
        os.environ.get("V2_MODEL", str(REPO / "99_runs/scaleworm_v2/weights/best.pt"))
    ),
    "v3": Path(
        os.environ.get("V3_MODEL", str(REPO / "99_runs/scaleworm_v3/weights/best.pt"))
    ),
}


def _day(fid: str) -> str:
    return fid.split("-")[1][:8]


def score_model(name, weights, frames, clicks, rng):
    """Return dict of aggregate metrics for one model over the gate frames."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    recs = []
    for fid, png in frames.items():
        pts = clicks[fid]
        res = model(str(png), conf=CONF, verbose=False)[0]
        preds = [
            (*b, c) for b, c in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist())
        ]
        tp, fp, fn = point_match(preds, pts)
        recs.append({"gt": len(pts), "pred": len(preds), "tp": tp, "fp": fp, "fn": fn})
    d = pd.DataFrame(recs)
    TP, FP, FN = d["tp"].sum(), d["fp"].sum(), d["fn"].sum()
    recall = TP / (TP + FN) if TP + FN else float("nan")
    prec = TP / (TP + FP) if TP + FP else float("nan")
    f1 = 2 * prec * recall / (prec + recall) if prec + recall else float("nan")
    fr_recall = np.array(
        [row["tp"] / row["gt"] for _, row in d.iterrows() if row["gt"]]
    )
    rec_ci = _boot_ci(fr_recall, np.mean, rng)
    bias = float((d["pred"] - d["gt"]).mean())
    mae = float((d["pred"] - d["gt"]).abs().mean())
    return {
        "model": name,
        "n": len(d),
        "gt_worms": int(d["gt"].sum()),
        "recall": recall,
        "recall_lo": rec_ci[0],
        "recall_hi": rec_ci[1],
        "precision": prec,
        "f1": f1,
        "count_bias": bias,
        "count_mae": mae,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "gate_dir",
        type=Path,
        help="unit gate dir (handcount_sheet.csv + clicks/ + frames/)",
    )
    ap.add_argument(
        "--keep-training", action="store_true", help="do NOT drop training-day frames"
    )
    ap.add_argument("--frames-dir", type=Path, default=None, help="override frames dir")
    args = ap.parse_args()

    gate = args.gate_dir
    sheet = pd.read_csv(gate / "handcount_sheet.csv")
    sheet = sheet[sheet["worm_count"].notna()]
    clicks_dir = gate / "clicks"
    frames_dir = args.frames_dir or (gate / "frames")
    tdays = set() if args.keep_training else training_days()

    frames, clicks, skipped = {}, {}, []
    for fid in sheet["frame_id"]:
        if _day(fid) in tdays:
            skipped.append(fid)
            continue
        png = frames_dir / f"{fid}.png"
        clk = clicks_dir / f"{fid}.json"
        if not png.exists() or not clk.exists():
            continue
        frames[fid] = png
        clicks[fid] = [tuple(p) for p in json.load(clk.open())["clicks"]]

    print(
        f"gate: {gate.name}  usable frames={len(frames)}  "
        f"(dropped {len(skipped)} training-day, leakage-safe)"
    )
    if not frames:
        raise SystemExit("no usable frames — check clicks/ and frames/")

    rng = np.random.default_rng(SEED)
    rows = [score_model(n, w, frames, clicks, rng) for n, w in MODELS.items()]
    tab = pd.DataFrame(rows)
    print(f"\n=== per-model gate result (conf={CONF}, seed={SEED}) ===")
    for r in rows:
        print(
            f"  {r['model']}: recall {r['recall']:.1%} "
            f"[{r['recall_lo']:.1%},{r['recall_hi']:.1%}]  "
            f"precision {r['precision']:.1%}  F1 {r['f1']:.1%}  "
            f"count bias {r['count_bias']:+.2f}  MAE {r['count_mae']:.2f}"
        )

    ok = [r for r in rows if r["recall"] >= MIN_RECALL]
    if ok:
        best = min(
            ok, key=lambda r: r["count_mae"]
        )  # best count agreement among adequate-recall
        corr = 1.0 / best["recall"]
        print(
            f"\nRECOMMEND: {best['model']} "
            f"(recall {best['recall']:.1%} ≥ {MIN_RECALL:.0%}, best count MAE {best['count_mae']:.2f}); "
            f"recall-correction ×{corr:.2f}"
        )
    else:
        top = max(rows, key=lambda r: r["recall"])
        print(
            f"\n⚠️ NO model clears MIN_RECALL={MIN_RECALL:.0%} "
            f"(best = {top['model']} at {top['recall']:.1%}). "
            "This unit needs a fine-tune or a low-confidence flag, not raw counts."
        )

    out = gate / "gate_result.csv"
    tab.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

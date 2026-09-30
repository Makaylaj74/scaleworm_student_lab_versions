"""Acceptance gate for the retrained detectors (v5_all / v5_clear).

Scores a MODEL on the held-out gate frames that were EXCLUDED (day-level) from the
Monday retrain (see build_retrain_dataset.py):
  - CLEAR  : validation/clear_window_handcount (39 frames) — independent CLICK ground truth
  - BLURRY : notebooks/model_comparison_handcount.csv (31 Feb+May-2024 frames) — hand counts;
             point-match uses the held-out MJ boxes (same frames, day-disjoint from training)

Reports two metrics per era so the result is both comparable and honest:
  1. COUNT-RATIO recall = sum(model)/sum(human)  — matches the published baselines
     (v2 clear 67.8%, v3 blurry ~33%); memory flags this as optimistic (FN cancels FP).
  2. POINT-MATCHED recall / precision / F1 (greedy point-in-box) — the honest detector metric.

Leakage is re-verified here against THIS model's actual training stems (not the old
hard-coded training days): asserts no gate frame shares a training day.

Usage (thesis venv, ultralytics 8.4.62, for comparability with baselines):
  MODEL=99_runs/scaleworm_v5_all/weights/best.pt LABEL=v5_all \
    /home/jovyan/joseph-scaleworm-thesis/.venv/bin/python3 scripts/gate_retrain.py
"""

from __future__ import annotations

import csv
import glob
import json
import os
import statistics
from pathlib import Path

REPO = Path("/home/jovyan/scaleworm-student-lab")
MONDAY = REPO / "validation" / "monday_manual_series"
CLEAR = REPO / "validation" / "clear_window_handcount"
BLURRY_SHEET = REPO / "notebooks" / "model_comparison_handcount.csv"

MODEL = os.environ.get("MODEL", str(REPO / "mushroom.pt"))
LABEL = os.environ.get("LABEL", Path(MODEL).stem)
OUT = Path(
    os.environ.get("OUT", str(REPO / "validation" / f"retrain_gate_{LABEL}.csv"))
)
CONF = float(os.environ.get("CONF", "0.25"))
BOOT_N = 10_000
SEED = 20260930


def day_of(stem: str) -> str:
    return stem.split("-")[1][:8]


def training_stems() -> set[str]:
    stems = set()
    for ds in ("scaleworm_retrain_all", "scaleworm_retrain_clear"):
        for p in glob.glob(str(REPO / "datasets" / ds / "labels" / "train" / "*.txt")):
            stems.add(Path(p).stem)
    return stems


def boot_ci(vals, n=BOOT_N, seed=SEED, lo=2.5, hi=97.5):
    import random

    rng = random.Random(seed)
    k = len(vals)
    if k == 0:
        return (float("nan"), float("nan"))
    out = []
    for _ in range(n):
        s = [vals[rng.randrange(k)] for _ in range(k)]
        out.append(sum(s) / len(s))
    out.sort()
    return out[int(lo / 100 * n)], out[int(hi / 100 * n)]


def point_match(preds, points):
    """preds: (x1,y1,x2,y2,conf); points: (px,py). Greedy point-in-box, conf desc."""
    claimed = [False] * len(points)
    tp = 0
    for x1, y1, x2, y2, _c in sorted(preds, key=lambda p: -p[4]):
        for j, (px, py) in enumerate(points):
            if not claimed[j] and x1 <= px <= x2 and y1 <= py <= y2:
                claimed[j] = True
                tp += 1
                break
    return tp, len(preds) - tp, len(points) - tp


def yolo_box_centers(stem: str, W: int, H: int):
    """MJ box centers (px) from the held-out Monday label file, for blurry point-match."""
    pts = []
    lbl = MONDAY / "labels" / f"{stem}.txt"
    for line in lbl.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        _, cx, cy, _, _ = line.split()
        pts.append((float(cx) * W, float(cy) * H))
    return pts


def report(era: str, recs: list[dict]) -> dict:
    n = len(recs)
    TP = sum(r["tp"] for r in recs)
    FP = sum(r["fp"] for r in recs)
    FN = sum(r["fn"] for r in recs)
    model_tot = sum(r["pred"] for r in recs)
    human_tot = sum(r["gt"] for r in recs)
    errs = [r["pred"] - r["gt"] for r in recs]
    count_recall = model_tot / human_tot if human_tot else 0.0
    det_recall = TP / (TP + FN) if TP + FN else float("nan")
    prec = TP / (TP + FP) if TP + FP else float("nan")
    f1 = (
        2 * prec * det_recall / (prec + det_recall)
        if (prec + det_recall)
        else float("nan")
    )
    fr_rec = [r["tp"] / r["gt"] for r in recs if r["gt"]]
    rlo, rhi = boot_ci(fr_rec)
    clo, chi = boot_ci([r["pred"] / r["gt"] for r in recs if r["gt"]])
    print(f"\n=== {LABEL} · {era}  (n={n}) ===")
    print(f"  human total/mean     : {human_tot} / {human_tot / n:.2f} per frame")
    print(f"  model total/mean     : {model_tot} / {model_tot / n:.2f} per frame")
    print(
        f"  COUNT-ratio recall   : {count_recall:.1%}  (per-frame 95% CI [{clo:.1%},{chi:.1%}])"
    )
    print(
        f"  POINT recall         : {det_recall:.1%}  (TP={TP} FN={FN}, per-frame 95% CI [{rlo:.1%},{rhi:.1%}])"
    )
    print(f"  POINT precision / F1 : {prec:.1%} (FP={FP}) / {f1:.1%}")
    print(
        f"  count MAE / bias     : {statistics.mean(abs(e) for e in errs):.2f} / {statistics.mean(errs):+.2f}"
    )
    return {
        "label": LABEL,
        "era": era,
        "n": n,
        "human": human_tot,
        "model": model_tot,
        "count_recall": round(count_recall, 4),
        "point_recall": round(det_recall, 4),
        "precision": round(prec, 4),
        "f1": round(f1, 4),
        "mae": round(statistics.mean(abs(e) for e in errs), 3),
        "bias": round(statistics.mean(errs), 3),
    }


def main() -> None:
    from ultralytics import YOLO

    tstems = training_stems()
    tdays = {day_of(s) for s in tstems}

    # --- gather gate rows ---
    clear_rows = [
        r
        for r in csv.DictReader(CLEAR.joinpath("handcount_sheet.csv").open())
        if r.get("frame_ready") == "Y" and (r.get("worm_count") or "").strip() != ""
    ]
    blurry_rows = [
        r
        for r in csv.DictReader(BLURRY_SHEET.open())
        if (r.get("human_count") or "").strip() not in ("", "nan")
    ]

    # --- leakage re-check against THIS retrain's actual training stems ---
    gate_ids = [r["frame_id"] for r in clear_rows] + [r["stem"] for r in blurry_rows]
    leaked = [g for g in gate_ids if day_of(g) in tdays]
    assert not leaked, f"LEAKAGE: gate frames share a training day: {leaked}"
    print(
        f"leakage OK: 0/{len(gate_ids)} gate frames on a training day "
        f"({len(tstems)} train stems, {len(tdays)} train days)"
    )

    model = YOLO(MODEL)
    summ = []

    # CLEAR: independent clicks as ground truth
    recs = []
    for r in clear_rows:
        fid = r["frame_id"]
        png = CLEAR / "frames" / f"{fid}.png"
        clk = CLEAR / "clicks" / f"{fid}.json"
        if not png.exists() or not clk.exists():
            print(f"  skip clear {fid}: missing frame/clicks")
            continue
        pts = [tuple(p) for p in json.load(clk.open())["clicks"]]
        res = model(str(png), conf=CONF, verbose=False)[0]
        preds = [
            (float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c))
            for b, c in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist())
        ]
        tp, fp, fn = point_match(preds, pts)
        recs.append(
            {
                "fid": fid,
                "gt": len(pts),
                "pred": len(preds),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )
    summ.append(report("CLEAR (clicks GT)", recs))

    # BLURRY: MJ held-out boxes as ground truth (centers), counts from hand sheet
    recs = []
    for r in blurry_rows:
        fid = r["stem"]
        png = MONDAY / "images" / f"{fid}.png"
        if not png.exists() or not (MONDAY / "labels" / f"{fid}.txt").exists():
            print(f"  skip blurry {fid}: missing frame/label")
            continue
        res = model(str(png), conf=CONF, verbose=False)[0]
        H, W = res.orig_shape
        pts = yolo_box_centers(fid, W, H)
        preds = [
            (float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c))
            for b, c in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist())
        ]
        tp, fp, fn = point_match(preds, pts)
        recs.append(
            {
                "fid": fid,
                "gt": len(pts),
                "pred": len(preds),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )
    summ.append(report("BLURRY (MJ-box GT)", recs))

    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows(summ)
    print(f"\nwrote {OUT}   (model={MODEL} conf={CONF} seed={SEED})")


if __name__ == "__main__":
    main()

"""v2/v3 detector recall on the held-out, leakage-free Monday val frames.

The Monday manual series was pre-labeled by v2 (clear) / v3 (blurry) and then
hand-corrected; those corrections are the ground truth here. This script re-runs the
routed detector on the val frames the models never saw and scores it against the
manual boxes.

Leakage control is DAY-LEVEL: v2/v3 were trained on 94 frames, 79 of which fall in the
Monday series. A val frame is dropped if ANY frame from its Monday (any 3-hourly slot)
was in the training set -- not just exact-id matches -- because same-day slots are
near-duplicate views of the same scene.

Matching is CENTER-BASED, not IoU: the manual click-boxes use a fixed median worm size,
so a prediction counts as a hit when its centre falls inside a manual box (greedy by
descending confidence, each manual box claimed once). Reported per routed model:
  * detection recall = TP / (TP + FN), precision = TP / (TP + FP), F1
  * count-level MAE, bias (pred - manual), ratio (pred / manual)
  * bootstrap 95% CI (percentile, n=10000, seed=20260913) over frames

Run on the GPU box (student-lab venv, L40S).
"""

from __future__ import annotations

import csv
import os
import sys
from datetime import date
from pathlib import Path

REPO = Path("/home/jovyan/scaleworm-student-lab")
sys.path.insert(0, str(REPO / "scripts"))

CONF = float(os.environ.get("CONF", "0.25"))
BLUR_ONSET = date(2023, 8, 10)
SEED = 20260913
N_BOOT = 10000
MANIFEST = REPO / "validation/monday_manual_series/frame_manifest.csv"
IMAGES = REPO / "validation/monday_manual_series/images"
LABELS = REPO / "validation/monday_manual_series/labels"
MODELS = {
    "v2": REPO / "99_runs/scaleworm_v2/weights/best.pt",
    "v3": REPO / "99_runs/scaleworm_v3/weights/best.pt",
}
TRAIN_LABEL_DIRS = [
    REPO / "datasets/scaleworm_v2/labels/train",
    REPO / "datasets/scaleworm_v2/labels/val",
]


def training_days() -> set[str]:
    """YYYYMMDD of every frame v2/v3 were trained on."""
    days = set()
    for d in TRAIN_LABEL_DIRS:
        for f in d.glob("*.txt"):
            days.add(f.stem.split("-")[1][:8])
    return days


def load_boxes_xyxy(path: Path, W: int, H: int) -> list[tuple[float, float, float, float]]:
    """Read a YOLO label file -> pixel xyxy boxes."""
    out = []
    if path.exists():
        for line in path.read_text().splitlines():
            p = line.split()
            if len(p) == 5:
                cx, cy, w, h = (float(v) for v in p[1:])
                cx, cy, w, h = cx * W, cy * H, w * W, h * H
                out.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return out


def center_match(preds, gts) -> tuple[int, int, int]:
    """Greedy centre-in-box match. preds: (x1,y1,x2,y2,conf) desc-sortable. gts: xyxy.

    Returns (tp, fp, fn). A prediction hits a GT if the prediction centre lies inside
    the GT box; each GT is claimed at most once, predictions processed by conf desc.
    """
    claimed = [False] * len(gts)
    tp = 0
    for x1, y1, x2, y2, _conf in sorted(preds, key=lambda p: -p[4]):
        pcx, pcy = (x1 + x2) / 2, (y1 + y2) / 2
        for j, (gx1, gy1, gx2, gy2) in enumerate(gts):
            if not claimed[j] and gx1 <= pcx <= gx2 and gy1 <= pcy <= gy2:
                claimed[j] = True
                tp += 1
                break
    fp = len(preds) - tp
    fn = len(gts) - tp
    return tp, fp, fn


def _boot_ci(values, stat, rng, n=N_BOOT):
    """Percentile 95% CI of `stat` over `values` via `n` bootstrap resamples."""
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(arr), size=(n, len(arr)))
    boots = np.array([stat(arr[i]) for i in idx])
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def main() -> None:
    import numpy as np
    from ultralytics import YOLO

    rng = np.random.default_rng(SEED)
    tdays = training_days()
    rows = list(csv.DictReader(MANIFEST.open(newline="")))
    val = [r for r in rows if r["split"] == "val"]
    clean = [r for r in val if r["frame_id"].split("-")[1][:8] not in tdays]
    print(f"val={len(val)}  clean(day-unseen)={len(clean)}  "
          f"excluded={len(val) - len(clean)}")

    models = {k: YOLO(str(v)) for k, v in MODELS.items()}
    per_model: dict[str, list[dict]] = {"v2": [], "v3": []}

    for r in clean:
        fid = r["frame_id"]
        route = r["prelabel_model"]
        png = IMAGES / f"{fid}.png"
        res = models[route](str(png), conf=CONF, verbose=False)[0]
        H, W = res.orig_shape
        preds = [
            (float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c))
            for b, c in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist())
        ]
        gts = load_boxes_xyxy(LABELS / f"{fid}.txt", W, H)
        tp, fp, fn = center_match(preds, gts)
        per_model[route].append(
            {"fid": fid, "gt": len(gts), "pred": len(preds), "tp": tp, "fp": fp, "fn": fn}
        )

    for m in ("v3", "v2"):  # v3 is the informative one (n large); v2 is n=2
        recs = per_model[m]
        n = len(recs)
        if n == 0:
            print(f"\n[{m}] no clean val frames.")
            continue
        TP = sum(x["tp"] for x in recs)
        FP = sum(x["fp"] for x in recs)
        FN = sum(x["fn"] for x in recs)
        gt = np.array([x["gt"] for x in recs], float)
        pred = np.array([x["pred"] for x in recs], float)
        recall = TP / (TP + FN) if TP + FN else float("nan")
        prec = TP / (TP + FP) if TP + FP else float("nan")
        f1 = 2 * prec * recall / (prec + recall) if prec + recall else float("nan")
        mae = np.mean(np.abs(pred - gt))
        bias = np.mean(pred - gt)
        # per-frame recall for a bootstrap CI (worm-weighted via resampling frames)
        fr_recall = np.array(
            [x["tp"] / x["gt"] if x["gt"] else np.nan for x in recs]
        )
        valid = fr_recall[~np.isnan(fr_recall)]
        rec_ci = _boot_ci(valid, np.mean, rng)
        mae_ci = _boot_ci(np.abs(pred - gt), np.mean, rng)
        print(f"\n[{m}] clean val frames n={n}  (manual worms={int(gt.sum())})")
        print(f"  detection recall = {recall:.2%}  (TP={TP} FN={FN})   "
              f"per-frame mean recall 95%CI [{rec_ci[0]:.2%}, {rec_ci[1]:.2%}]")
        print(f"  precision        = {prec:.2%}  (FP={FP})   F1={f1:.2%}")
        print(f"  count MAE        = {mae:.2f} worms/frame  95%CI "
              f"[{mae_ci[0]:.2f}, {mae_ci[1]:.2f}]")
        print(f"  count bias       = {bias:+.2f} (pred - manual)   "
              f"model total {int(pred.sum())} vs manual {int(gt.sum())} "
              f"({pred.sum() / gt.sum():.1%})")


if __name__ == "__main__":
    main()

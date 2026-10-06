"""Gate the tiled detector (v8) under SAHI against v5_all, on dense + regression test sets.

Answers: does training on tiles + SAHI inference beat (a) v5_all plain and (b) v5_all wrapped in
SAHI-at-inference (the earlier dead end), on DENSE frames -- without regressing the clear/early gates?

Compares three configurations on each stratum, point-matched (greedy centre-in-box, same metric as
the other gates), reporting count-ratio / point-recall / precision:
  * v5_all  plain   (full-frame inference, the production baseline)
  * v5_all  + SAHI  (inference-only tiling -- the prior underwhelming result)
  * v8_tiled + SAHI (trained AND inferred on tiles -- the new approach)

Strata (GT in parentheses):
  dense_clear   held-out dense clear Monday frames   (box GT, >=18 worms, not in v5/tiled train)
  dense_blurry  dense Tuesday frames                 (box GT, >=25 worms)
  clear39       clear-window handcount               (click GT) -- regression check
  janaug        early-2021 handcount                 (click GT) -- the 2021 target

Run (thesis venv, has sahi + cuda):
  STRATUM=dense_clear /home/jovyan/joseph-scaleworm-thesis/.venv/bin/python scripts/gate_tiled_sahi.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from PIL import Image

REPO = Path("/home/jovyan/scaleworm-student-lab")
V5 = REPO / "99_runs/scaleworm_v5_all/weights/best.pt"
V8 = REPO / "99_runs/scaleworm_v8_tiled/weights/best.pt"
CONF = float(os.environ.get("CONF", "0.40"))
SLICE = int(os.environ.get("SLICE", "640"))
OVERLAP = float(os.environ.get("OVERLAP", "0.2"))
STRATUM = os.environ.get("STRATUM", "dense_clear")


def point_match(preds, points):
    """preds (x1,y1,x2,y2,conf) px; points (px,py). Greedy point-in-box, conf desc."""
    claimed = [False] * len(points)
    tp = 0
    for x1, y1, x2, y2, _c in sorted(preds, key=lambda p: -p[4]):
        for j, (px, py) in enumerate(points):
            if not claimed[j] and x1 <= px <= x2 and y1 <= py <= y2:
                claimed[j] = True
                tp += 1
                break
    return tp, len(preds) - tp, len(points) - tp


def box_centers(lbl: Path, w: int, h: int):
    pts = []
    for ln in lbl.read_text().splitlines():
        if ln.strip():
            _, cx, cy, _, _ = ln.split()
            pts.append((float(cx) * w, float(cy) * h))
    return pts


def frames_and_gt(stratum: str):
    """Yield (image_path, [gt points in px]) for the chosen stratum."""
    if stratum == "clear39":
        d = REPO / "validation/clear_window_handcount"
        for jp in sorted((d / "clicks").glob("*.json")):
            dat = json.loads(jp.read_text())
            img = d / "frames" / f"{dat['frame_id']}.png"
            if img.exists():
                yield img, [(float(x), float(y)) for x, y in dat["clicks"]]
    elif stratum == "janaug":
        d = REPO / "validation/early_2021_handcount"
        for jp in sorted((d / "clicks").glob("*.json")):
            dat = json.loads(jp.read_text())
            img = d / "frames" / f"{dat['frame_id']}.png"
            if img.exists():
                yield img, [(float(x), float(y)) for x, y in dat["clicks"]]
    else:  # dense_clear / dense_blurry -> box GT, held out from v5 train
        if stratum == "dense_clear":
            src, thr, clear_only = REPO / "validation/monday_manual_series", 18, True
        else:
            src, thr, clear_only = REPO / "validation/tuesday_manual_series", 25, False
        train = {p.stem for p in (REPO / "datasets/scaleworm_retrain_all/images/train").glob("*")}
        for lbl in sorted((src / "labels").glob("*.txt")):
            stem = lbl.stem
            if stem in train:
                continue
            if clear_only and not stem.split("-")[1][:8] < "20230811":
                continue
            if sum(1 for ln in lbl.read_text().splitlines() if ln.strip()) < thr:
                continue
            img = next((src / "images").glob(f"{stem}.*"), None)
            if img:
                w, h = Image.open(img).size
                yield img, box_centers(lbl, w, h)


def plain_fn(model):
    def f(img):
        r = model(str(img), conf=CONF, imgsz=1280, verbose=False)[0]
        return [(a, b, c, d, s) for (a, b, c, d), s in
                zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]
    return f


def sahi_fn(det):
    from sahi.predict import get_sliced_prediction

    def f(img):
        r = get_sliced_prediction(
            str(img), det, slice_height=SLICE, slice_width=SLICE,
            overlap_height_ratio=OVERLAP, overlap_width_ratio=OVERLAP, verbose=0,
        )
        return [(p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy, p.score.value)
                for p in r.object_prediction_list]
    return f


def score(label, pred_fn, data):
    gt = pred = tp = fp = fn = 0
    for img, pts in data:
        preds = pred_fn(img)
        a, b, c = point_match(preds, pts)
        tp += a
        fp += b
        fn += c
        gt += len(pts)
        pred += len(preds)
    rec = tp / (tp + fn) if tp + fn else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    print(f"  {label:18s} human={gt:5d} model={pred:5d}  count={pred / gt if gt else 0:6.1%}  "
          f"point-recall={rec:6.1%}  precision={prec:6.1%}")


def main() -> None:
    from sahi import AutoDetectionModel
    from ultralytics import YOLO

    data = list(frames_and_gt(STRATUM))
    print(f"\n=== STRATUM={STRATUM}  frames={len(data)}  conf={CONF} slice={SLICE}/{OVERLAP} ===")

    v5 = YOLO(str(V5))
    score("v5_all plain", plain_fn(v5), data)
    v5s = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=str(V5), confidence_threshold=CONF, device="cuda:0")
    score("v5_all +SAHI", sahi_fn(v5s), data)
    if V8.exists():
        v8s = AutoDetectionModel.from_pretrained(
            model_type="ultralytics", model_path=str(V8), confidence_threshold=CONF, device="cuda:0")
        score("v8_tiled +SAHI", sahi_fn(v8s), data)
    else:
        print("  (v8_tiled weights not found yet)")


if __name__ == "__main__":
    main()

"""Acceptance gate for the v7_2021 fine-tune: score a MODEL on the 4 clicked Jan-Aug 2021
front-on frames (validation/early_2021_handcount) that are the ONLY independent GT in the
early-2021 undercount region. All 4 are day-disjoint from v7_2021 training (the 0216/0617
same-day training frames were quarantined to leakage_excluded_2021).

Reports count-ratio recall + greedy point-in-box recall/precision, per-frame and pooled, so
v5_all (prior 22/94 = 0.23) and v7_2021 are directly comparable. Run both in the thesis venv
(ultralytics 8.4.62) for baseline comparability:

  MODEL=99_runs/scaleworm_v5_all/weights/best.pt  LABEL=v5_all  CONF=0.40 \
    /home/jovyan/joseph-scaleworm-thesis/.venv/bin/python3 scripts/gate_2021_janaug.py
  MODEL=99_runs/scaleworm_v7_2021/weights/best.pt LABEL=v7_2021 CONF=0.40 \
    /home/jovyan/joseph-scaleworm-thesis/.venv/bin/python3 scripts/gate_2021_janaug.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ultralytics import YOLO

REPO = Path("/home/jovyan/scaleworm-student-lab")
GATE = REPO / "validation" / "early_2021_handcount"

MODEL = os.environ.get("MODEL", str(REPO / "99_runs/scaleworm_v7_2021/weights/best.pt"))
LABEL = os.environ.get("LABEL", Path(MODEL).stem)
CONF = float(os.environ.get("CONF", "0.40"))
IMGSZ = int(os.environ.get("IMGSZ", "1280"))


def point_match(preds, points):
    """preds: (x1,y1,x2,y2,conf); points: (px,py). Greedy point-in-box, conf desc.

    Verbatim from gate_retrain.py so the metric matches the committed v5 gate.
    """
    claimed = [False] * len(points)
    tp = 0
    for x1, y1, x2, y2, _c in sorted(preds, key=lambda p: -p[4]):
        for j, (px, py) in enumerate(points):
            if not claimed[j] and x1 <= px <= x2 and y1 <= py <= y2:
                claimed[j] = True
                tp += 1
                break
    return tp, len(preds) - tp, len(points) - tp


def main() -> None:
    model = YOLO(MODEL)
    clicks_dir = GATE / "clicks"
    rows = []
    for jp in sorted(clicks_dir.glob("*.json")):
        d = json.loads(jp.read_text())
        points = [(float(x), float(y)) for x, y in d["clicks"]]
        png = GATE / "frames" / f"{d['frame_id']}.png"
        res = model(str(png), conf=CONF, imgsz=IMGSZ, verbose=False)[0]
        preds = [
            (b[0], b[1], b[2], b[3], c)
            for b, c in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist())
        ]
        tp, fp, fn = point_match(preds, points)
        rows.append(
            {
                "stem": d["frame_id"],
                "gt": len(points),
                "pred": len(preds),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )

    human = sum(r["gt"] for r in rows)
    pred = sum(r["pred"] for r in rows)
    TP = sum(r["tp"] for r in rows)
    FP = sum(r["fp"] for r in rows)
    FN = sum(r["fn"] for r in rows)
    print(
        f"\n=== {LABEL} · Jan-Aug 2021 (n={len(rows)})  conf={CONF} imgsz={IMGSZ} ==="
    )
    for r in rows:
        cr = r["pred"] / r["gt"] if r["gt"] else 0.0
        pr = r["tp"] / r["gt"] if r["gt"] else 0.0
        print(
            f"  {r['stem']}: human {r['gt']:3d}  model {r['pred']:3d}  "
            f"count {cr:5.0%}  point-recall {pr:5.0%}"
        )
    print(
        f"  --- pooled: human {human}  model {pred}  "
        f"count-ratio {pred / human:.1%}  "
        f"point-recall {TP / (TP + FN):.1%}  "
        f"precision {TP / (TP + FP) if TP + FP else 0:.1%}"
    )


if __name__ == "__main__":
    main()

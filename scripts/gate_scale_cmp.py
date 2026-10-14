"""Point-match gate for the YOLO26 scale sweep: does flat val-mAP hold under the lab metric?

The scale sweep (scripts/train_scale_cmp_yolo26.py) showed val mAP ~flat across n/s/m/l/x.
val mAP is not the lab's detection metric, so this confirms (or refutes) that on the real
click/box GT using the SAME greedy centre-in-box point-match as the other gates.

Compares yolo26 {n, s, x} (same recipe -- COCO-pretrained, fine-tuned on retrain_all, differ
ONLY in scale) PLAIN at conf 0.40 / imgsz 1280 on:
  clear39      clear-window handcount  (click GT)  -- the main clear-window deliverable
  dense_clear  held-out dense Monday   (box GT, >=18 worms, excluded from retrain_all train)

Plain (no SAHI) so the scale comparison is internally clean; production clear uses +SAHI, so
these absolute recalls run a few pts below the quoted v8+SAHI numbers -- read the s-vs-x GAP,
not the level. Reuses the point_match / frames_and_gt / plain_fn helpers from gate_tiled_sahi.

Run (scaleworm-student-lab venv, ultralytics 8.4.131 + cuda):
  /home/jovyan/scaleworm-student-lab/.venv/bin/python scripts/gate_scale_cmp.py
Env overrides: SCALES=n,s,x  STRATA=clear39,dense_clear  CONF=0.40
"""

from __future__ import annotations

import os
from pathlib import Path

from gate_tiled_sahi import frames_and_gt, plain_fn, score

REPO = Path("/home/jovyan/scaleworm-student-lab")
SCALES = os.environ.get("SCALES", "n,s,x").split(",")
STRATA = os.environ.get("STRATA", "clear39,dense_clear").split(",")


def main() -> None:
    from ultralytics import YOLO

    models = {
        s: YOLO(str(REPO / f"99_runs/scale_cmp_yolo26{s}/weights/best.pt"))
        for s in SCALES
    }
    for stratum in STRATA:
        data = list(frames_and_gt(stratum))
        print(
            f"\n=== STRATUM={stratum}  frames={len(data)}  conf={os.environ.get('CONF', '0.40')} ==="
        )
        for s in SCALES:
            score(f"yolo26{s} plain", plain_fn(models[s]), data)


if __name__ == "__main__":
    main()

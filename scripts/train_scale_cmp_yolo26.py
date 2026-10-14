"""Train a YOLO26 scale sweep (n/s/m/l/x) on the scaleworm detector dataset.

Purpose: the production detector is YOLO26-x (verified 2026-10-14: every checkpoint,
mushroom.pt included, embeds yolo26x.yaml -- YOLO11 was never used). This sweep asks
whether a smaller/faster YOLO26 scale holds detection quality on scaleworm frames, which
matters for the ~13k-record M-F x 2015-2024 harvest where inference cost is the wall.

Design (identical across all 5 models -- only the scale varies):
  * data   : datasets/scaleworm_retrain_all (421 train / 87 val, 1280px full-frame) --
             the SAME set the production v5_all was trained on, so the ranking maps onto
             the incumbent.
  * init   : each scale from its own COCO-pretrained yolo26{scale}.pt (fair -- mushroom.pt
             only exists at x, so warm-starting from it would privilege x).
  * hparams: imgsz 1280, epochs 100, batch 16, seed 0, workers 8, deterministic.

Reports per scale: params, val mAP50 / mAP50-95 / precision / recall (ultralytics val on the
shared retrain_all val split) + inference ms/img. This is a scale-RANKING on a shared val
split -- NOT the lab's independent click-validated point-match gate (see scripts/gate_tiled_sahi.py);
run that on the winner afterwards for the faithful detection-recall number.

Run (scaleworm-student-lab venv, has cuda + headless opencv):
  /home/jovyan/scaleworm-student-lab/.venv/bin/python scripts/train_scale_cmp_yolo26.py
Env overrides: SCALES=n,s,m,l,x  EPOCHS=100  IMGSZ=1280  BATCH=16  WORKERS=8  DEVICE=0
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

REPO = Path("/home/jovyan/scaleworm-student-lab")
DATA = REPO / "datasets/scaleworm_retrain_all/data_resolved.yaml"
PROJECT = REPO / "99_runs"
OUT_CSV = REPO / "99_runs/scale_cmp_yolo26_results.csv"

SCALES = os.environ.get("SCALES", "n,s,m,l,x").split(",")
EPOCHS = int(os.environ.get("EPOCHS", "100"))
IMGSZ = int(os.environ.get("IMGSZ", "1280"))
BATCH = int(os.environ.get("BATCH", "16"))
WORKERS = int(os.environ.get("WORKERS", "8"))  # <=24 (lab cap); retrain_all is small
DEVICE = os.environ.get("DEVICE", "0")
SEED = int(os.environ.get("SEED", "0"))


def train_one(scale: str) -> dict:
    """Fine-tune one YOLO26 scale and return its val metrics + inference speed."""
    from ultralytics import YOLO

    name = f"scale_cmp_yolo26{scale}"
    model = YOLO(f"yolo26{scale}.pt")  # COCO-pretrained init for this scale
    n_params = sum(p.numel() for p in model.model.parameters())

    model.train(
        data=str(DATA),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        seed=SEED,
        deterministic=True,
        project=str(PROJECT),
        name=name,
        exist_ok=True,
        verbose=False,
    )

    # Final val on the shared retrain_all val split (best.pt is auto-loaded post-train).
    metrics = model.val(
        data=str(DATA), imgsz=IMGSZ, batch=BATCH, device=DEVICE, verbose=False
    )
    speed = metrics.speed.get("inference", float("nan"))  # ms/img
    return {
        "scale": scale,
        "params_M": round(n_params / 1e6, 2),
        "map50": round(float(metrics.box.map50), 4),
        "map50_95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "infer_ms": round(float(speed), 2),
    }


def main() -> None:
    rows: list[dict] = []
    for scale in SCALES:
        print(f"\n===== training yolo26{scale} =====", flush=True)
        try:
            row = train_one(scale)
            rows.append(row)
            print(f"DONE yolo26{scale}: {row}", flush=True)
        except Exception as e:  # noqa: BLE001 -- keep sweep going if one scale fails
            print(f"FAILED yolo26{scale}: {type(e).__name__}: {e}", flush=True)

    if rows:
        with OUT_CSV.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {OUT_CSV} ({len(rows)} scales)", flush=True)


if __name__ == "__main__":
    main()

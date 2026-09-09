"""Fine-tune mushroom.pt -> scaleworm v2 (domain adaptation to new camera units).

Starts from the inherited mushroom.pt weights and fine-tunes on the CORRECTED labels in
datasets/scaleworm_v2/images|labels/{train,val}. After training, evaluate the run on the
independent 39-frame clear-window gate with scripts/run_recall_gate.py (point MODEL at the
new best.pt) — that, not the training-val mAP, is the acceptance test.

COMPUTE NOTE: this Hub's GPU driver (CUDA 12020) is too old for the installed torch, so
training falls back to CPU and full-res training is impractically slow. Either train at a
reduced imgsz / few epochs as a smoke test here, or run this on a GPU box (Colab, a lab
workstation) with the same dataset. Set DEVICE below.

    python scripts/train_v2.py
"""

from __future__ import annotations

import os
from pathlib import Path

from ultralytics import YOLO

REPO = Path("/home/jovyan/scaleworm-student-lab")

# --- config (env-overridable so a CPU smoke test is reproducible without editing) ---
# Full-GPU defaults; override for a CPU smoke test, e.g.:
#   IMGSZ=640 EPOCHS=60 RUN_NAME=scaleworm_v2_smoke python scripts/train_v2.py
BASE = REPO / "mushroom.pt"
DATA = REPO / "datasets/scaleworm_v2/data.yaml"
EPOCHS = int(os.environ.get("EPOCHS", "100"))
IMGSZ = int(os.environ.get("IMGSZ", "1280"))  # small worms; 1280 detail, 640 for CPU
BATCH = int(os.environ.get("BATCH", "8"))
DEVICE = os.environ.get("DEVICE", "cpu")  # set to "0" on a machine with a supported GPU
FREEZE = int(os.environ.get("FREEZE", "10"))  # freeze backbone; 0 = train all layers
WORKERS = int(os.environ.get("WORKERS", "8"))  # dataloader workers (lab cap: keep <=24)
PATIENCE = int(os.environ.get("PATIENCE", "20"))
RUN_NAME = os.environ.get("RUN_NAME", "scaleworm_v2")


def main() -> None:
    model = YOLO(str(BASE))
    model.train(
        data=str(DATA),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        freeze=FREEZE,
        workers=WORKERS,
        patience=PATIENCE,
        project=str(REPO / "99_runs"),
        name=RUN_NAME,
        seed=20260908,
        exist_ok=True,
    )
    print(f"done -> weights under 99_runs/{RUN_NAME}/weights/best.pt")
    print("validate: edit MODEL in run_recall_gate.py to that best.pt and re-run the gate")


if __name__ == "__main__":
    main()

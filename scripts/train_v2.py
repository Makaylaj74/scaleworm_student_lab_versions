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

from pathlib import Path

from ultralytics import YOLO

REPO = Path("/home/jovyan/scaleworm-student-lab")

# --- config (tune when real GPU is available) ---
BASE = REPO / "mushroom.pt"
DATA = REPO / "datasets/scaleworm_v2/data.yaml"
EPOCHS = 100
IMGSZ = 1280  # worms are small; 1280 preserves detail (drop to 640 for a CPU smoke test)
BATCH = 8
DEVICE = "cpu"  # set to 0 on a machine with a supported GPU
FREEZE = 10  # freeze backbone layers for stable fine-tuning; set 0 to train all


def main() -> None:
    model = YOLO(str(BASE))
    model.train(
        data=str(DATA),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        freeze=FREEZE,
        patience=20,
        project=str(REPO / "99_runs"),
        name="scaleworm_v2",
        seed=20260908,
        exist_ok=True,
    )
    print("done -> weights under 99_runs/scaleworm_v2/weights/best.pt")
    print("validate: edit MODEL in run_recall_gate.py to that best.pt and re-run the gate")


if __name__ == "__main__":
    main()

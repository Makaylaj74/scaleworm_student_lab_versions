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

import yaml
from ultralytics import YOLO

# Resolve repo root from this file so the script runs on any machine (e.g. a GPU box),
# not just the Hub. Override BASE/DATA via env if the layout differs.
REPO = Path(__file__).resolve().parents[1]

# --- config (env-overridable so a CPU smoke test is reproducible without editing) ---
# Full-GPU defaults; override for a CPU smoke test, e.g.:
#   IMGSZ=640 EPOCHS=60 RUN_NAME=scaleworm_v2_smoke python scripts/train_v2.py
BASE = Path(os.environ.get("BASE", str(REPO / "mushroom.pt")))
DATA = Path(os.environ.get("DATA", str(REPO / "datasets/scaleworm_v2/data.yaml")))
EPOCHS = int(os.environ.get("EPOCHS", "100"))
IMGSZ = int(os.environ.get("IMGSZ", "1280"))  # small worms; 1280 detail, 640 for CPU
BATCH = int(os.environ.get("BATCH", "8"))
DEVICE = os.environ.get("DEVICE", "cpu")  # set to "0" on a machine with a supported GPU
FREEZE = int(os.environ.get("FREEZE", "10"))  # freeze backbone; 0 = train all layers
WORKERS = int(os.environ.get("WORKERS", "8"))  # dataloader workers (lab cap: keep <=24)
PATIENCE = int(os.environ.get("PATIENCE", "20"))
RUN_NAME = os.environ.get("RUN_NAME", "scaleworm_v2")
# Initial learning rate. Unset = ultralytics default (0.01). Set LOW (e.g. 0.001) for a
# gentle warm-start fine-tune from an already-good model (e.g. BASE=v2) so the new domain
# is learned without forgetting the old one. NOTE: ultralytics IGNORES lr0 under the
# default optimizer=auto, so when LR0 is set we pin an explicit optimizer (OPTIMIZER, or
# SGD) so the requested lr0 actually takes effect.
LR0 = os.environ.get("LR0")
OPTIMIZER = os.environ.get(
    "OPTIMIZER"
)  # e.g. "SGD"/"AdamW"; default auto unless LR0 set


def _resolved_data() -> str:
    """Write a copy of data.yaml with `path` pinned to this machine's dataset dir.

    Lets the same committed data.yaml train on any host (Hub or a GPU box) without
    hand-editing the absolute `path`. Output is gitignored (data_resolved.yaml).
    """
    cfg = yaml.safe_load(DATA.read_text())
    cfg["path"] = str(DATA.parent)
    out = DATA.parent / "data_resolved.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return str(out)


def main() -> None:
    model = YOLO(str(BASE))
    train_kwargs = dict(
        data=_resolved_data(),
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
    if LR0 is not None:
        train_kwargs["lr0"] = float(LR0)
        train_kwargs["optimizer"] = OPTIMIZER or "SGD"  # auto ignores lr0
    elif OPTIMIZER:
        train_kwargs["optimizer"] = OPTIMIZER
    print(
        f"BASE={BASE.name}  LR0={LR0 or 'default'}  "
        f"OPTIMIZER={train_kwargs.get('optimizer', 'auto')}  FREEZE={FREEZE}  RUN_NAME={RUN_NAME}"
    )
    model.train(**train_kwargs)
    print(f"done -> weights under 99_runs/{RUN_NAME}/weights/best.pt")
    print(
        "validate: edit MODEL in run_recall_gate.py to that best.pt and re-run the gate"
    )


if __name__ == "__main__":
    main()

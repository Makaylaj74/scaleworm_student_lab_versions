"""Seed replication for the YOLO26 scale comparison (s vs x).

The single-seed gate (scripts/gate_scale_cmp.py) showed yolo26s beating yolo26x by ~6 pts
point-recall on clear39 + dense_clear. Per the lab's Defensible-Statistics rule, a gap from
n=1 run each is an anecdote -- this retrains s and x at additional seeds so the gap can be
reported as mean +/- spread and tested against seed variance.

Trains {SCALES} x {SEEDS} with hyperparams IDENTICAL to train_scale_cmp_yolo26.py (the seed-0
run) -- only the seed changes. Run dirs: scale_cmp_yolo26{scale}_seed{seed}. Seed 0 already
exists as scale_cmp_yolo26{scale} (reused by the aggregating gate, not retrained here).

Run (scaleworm-student-lab venv, cuda):
  /home/jovyan/scaleworm-student-lab/.venv/bin/python scripts/train_scale_seeds.py
Env overrides: SCALES=s,x  SEEDS=1,2  EPOCHS=100  IMGSZ=1280  BATCH=16  WORKERS=8  DEVICE=0
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path("/home/jovyan/scaleworm-student-lab")
DATA = REPO / "datasets/scaleworm_retrain_all/data_resolved.yaml"
PROJECT = REPO / "99_runs"

SCALES = os.environ.get("SCALES", "s,x").split(",")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "1,2").split(",")]
EPOCHS = int(os.environ.get("EPOCHS", "100"))
IMGSZ = int(os.environ.get("IMGSZ", "1280"))
BATCH = int(os.environ.get("BATCH", "16"))
WORKERS = int(os.environ.get("WORKERS", "8"))  # <=24 (lab cap)
DEVICE = os.environ.get("DEVICE", "0")


def train_one(scale: str, seed: int) -> None:
    from ultralytics import YOLO

    name = f"scale_cmp_yolo26{scale}_seed{seed}"
    print(f"\n===== training yolo26{scale} seed={seed} -> {name} =====", flush=True)
    model = YOLO(f"yolo26{scale}.pt")
    model.train(
        data=str(DATA),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        seed=seed,
        deterministic=True,
        project=str(PROJECT),
        name=name,
        exist_ok=True,
        verbose=False,
    )
    print(f"DONE {name}", flush=True)


def main() -> None:
    for scale in SCALES:
        for seed in SEEDS:
            try:
                train_one(scale, seed)
            except Exception as e:  # noqa: BLE001 -- keep the sweep going
                print(
                    f"FAILED yolo26{scale} seed={seed}: {type(e).__name__}: {e}",
                    flush=True,
                )


if __name__ == "__main__":
    main()

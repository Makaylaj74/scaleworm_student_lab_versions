"""Recall gate: mushroom.pt vs hand counts on the clear-window validation frames.

Runs the inherited detector on the hand-counted clear-window frames and compares
model counts to Makayla's click-counts, to decide whether the existing detector works
on sharp footage (the gate) before investing in retraining. Mirrors the earlier
Jan-2023 / Feb-2024 / May-2024 vs-human comparisons (same model, same conf=0.25).
Reports overall and broken out by camera unit (-2021 / -2022), since the clear window
spans both clear units and we want recall to hold on each.

Run under the thesis venv (ultralytics 8.4.62) for comparability with those numbers:
    /home/jovyan/joseph-scaleworm-thesis/.venv/bin/python scripts/run_recall_gate.py
"""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

from ultralytics import YOLO

REPO = Path("/home/jovyan/scaleworm-student-lab")
sys.path.insert(0, str(REPO / "scripts"))
from count_frames import count_worms

VAL = REPO / "validation" / "clear_window_handcount"
SHEET = VAL / "handcount_sheet.csv"
FRAMES = VAL / "frames"
MODEL = REPO / "mushroom.pt"
OUT = VAL / "recall_gate_results.csv"
CONF = 0.25
BOOT_N = 10_000
SEED = 20260908


def bootstrap_ci(pairs, stat_fn, n=BOOT_N, seed=SEED, lo=2.5, hi=97.5):
    """Percentile bootstrap CI over frames. pairs = [(model, human), ...]."""
    import random

    rng = random.Random(seed)
    k = len(pairs)
    vals = []
    for _ in range(n):
        sample = [pairs[rng.randrange(k)] for _ in range(k)]
        vals.append(stat_fn(sample))
    vals.sort()
    return vals[int(lo / 100 * n)], vals[int(hi / 100 * n)]


def report(label: str, pairs: list[tuple[int, int]]) -> None:
    """Print a recall summary block for a set of (model, human) pairs."""
    model_tot = sum(m for m, _ in pairs)
    human_tot = sum(h for _, h in pairs)
    errs = [m - h for m, h in pairs]
    recall = model_tot / human_tot if human_tot else 0.0
    mae = statistics.mean(abs(e) for e in errs)
    bias = statistics.mean(errs)
    rmse = statistics.mean(e * e for e in errs) ** 0.5
    exact = sum(1 for m, h in pairs if m == h) / len(pairs)
    mae_lo, mae_hi = bootstrap_ci(pairs, lambda s: statistics.mean(abs(m - h) for m, h in s))
    rec_lo, rec_hi = bootstrap_ci(
        pairs, lambda s: sum(m for m, _ in s) / max(1, sum(h for _, h in s))
    )
    print(f"\n=== {label}  (n={len(pairs)}) ===")
    print(f"  human total / mean   : {human_tot} / {human_tot / len(pairs):.2f} per frame")
    print(f"  model total / mean   : {model_tot} / {model_tot / len(pairs):.2f} per frame")
    print(f"  RECALL (model/human) : {recall:.1%}   95% CI [{rec_lo:.1%}, {rec_hi:.1%}]")
    print(f"  MAE                  : {mae:.2f}   95% CI [{mae_lo:.2f}, {mae_hi:.2f}]")
    print(f"  bias (model-human)   : {bias:+.2f}   RMSE {rmse:.2f}   exact {exact:.0%}")


def main() -> None:
    rows = [
        r
        for r in csv.DictReader(SHEET.open())
        if r.get("frame_ready") == "Y" and r["worm_count"].strip() != ""
    ]
    model = YOLO(str(MODEL))
    names = model.names

    results = []
    for r in rows:
        fid = r["frame_id"]
        human = int(r["worm_count"])
        res = model(str(FRAMES / f"{fid}.png"), conf=CONF, verbose=False)[0]
        cls_ids = [int(c) for c in res.boxes.cls.tolist()]
        m = count_worms(cls_ids, names)
        results.append((fid, r.get("camera_unit", ""), m, human))

    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_id", "camera_unit", "model_count", "human_count", "error"])
        for fid, unit, m, h in results:
            w.writerow([fid, unit, m, h, m - h])
    with OUT.open("a") as f:
        f.write(
            f"# model=mushroom.pt conf={CONF} ultralytics=8.4.62 "
            f"n={len(results)} seed={SEED} boot={BOOT_N}\n"
        )

    report("OVERALL clear window", [(m, h) for _, _, m, h in results])
    for unit in sorted({u for _, u, _, _ in results}):
        report(unit, [(m, h) for _, u, m, h in results if u == unit])
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

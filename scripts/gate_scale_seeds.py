"""Aggregate the YOLO26 s-vs-x point-match gate across seeds (mean +/- spread).

Scores yolo26{s,x} at seeds {0,1,2} on clear39 + dense_clear with the lab point-match metric,
then reports per-scale mean +/- sd point-recall and the s-minus-x gap vs its spread -- so the
single-seed "s beats x by ~6 pts" can be judged real or noise (lab Defensible-Statistics rule).

Seed 0 weights = 99_runs/scale_cmp_yolo26{scale}; seeds 1,2 = ..._seed{N} (train_scale_seeds.py).

Run (scaleworm-student-lab venv, cuda):
  /home/jovyan/scaleworm-student-lab/.venv/bin/python scripts/gate_scale_seeds.py
Env overrides: SCALES=s,x  SEEDS=0,1,2  STRATA=clear39,dense_clear  CONF=0.40
"""

from __future__ import annotations

import os
from pathlib import Path
from statistics import mean, pstdev

from gate_tiled_sahi import frames_and_gt, plain_fn, point_match

REPO = Path("/home/jovyan/scaleworm-student-lab")
SCALES = os.environ.get("SCALES", "s,x").split(",")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
STRATA = os.environ.get("STRATA", "clear39,dense_clear").split(",")


def weights(scale: str, seed: int) -> Path:
    name = (
        f"scale_cmp_yolo26{scale}"
        if seed == 0
        else f"scale_cmp_yolo26{scale}_seed{seed}"
    )
    return REPO / f"99_runs/{name}/weights/best.pt"


def recall_of(pred_fn, data) -> float:
    tp = fn = 0
    for img, pts in data:
        a, _b, c = point_match(pred_fn(img), pts)
        tp += a
        fn += c
    return tp / (tp + fn) if tp + fn else 0.0


def main() -> None:
    from ultralytics import YOLO

    # cache data per stratum
    strata_data = {st: list(frames_and_gt(st)) for st in STRATA}
    # recalls[stratum][scale] = [per-seed recall]
    recalls: dict[str, dict[str, list[float]]] = {
        st: {s: [] for s in SCALES} for st in STRATA
    }

    for scale in SCALES:
        for seed in SEEDS:
            w = weights(scale, seed)
            if not w.exists():
                print(f"  (missing {w} -- skipping)", flush=True)
                continue
            m = YOLO(str(w))
            for st in STRATA:
                recalls[st][scale].append(recall_of(plain_fn(m), strata_data[st]))

    for st in STRATA:
        print(
            f"\n=== {st}  (frames={len(strata_data[st])}, conf={os.environ.get('CONF', '0.40')}) ==="
        )
        stats = {}
        for scale in SCALES:
            rs = recalls[st][scale]
            if rs:
                mu, sd = mean(rs), (pstdev(rs) if len(rs) > 1 else 0.0)
                stats[scale] = (mu, sd)
                seeds_str = ", ".join(f"{r:.1%}" for r in rs)
                print(
                    f"  yolo26{scale}: recall {mu:.1%} +/- {sd:.1%}  (seeds: {seeds_str})"
                )
        if "s" in stats and "x" in stats:
            gap = stats["s"][0] - stats["x"][0]
            spread = stats["s"][1] + stats["x"][1]
            verdict = (
                "ROBUST (gap > combined sd)"
                if abs(gap) > spread
                else "WITHIN NOISE (gap <= combined sd)"
            )
            print(f"  --> s - x = {gap:+.1%}  (combined sd {spread:.1%})  => {verdict}")


if __name__ == "__main__":
    main()

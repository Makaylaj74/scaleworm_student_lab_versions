"""Does higher-resolution inference fix v5_all's high-density under-count?

v5_all compresses dense clear frames (recovers ~64% at >22 worms/frame). One candidate
real fix (no post-hoc curve fitting): run inference at a larger imgsz so small worms in
crowded frames get more pixels. This tests v5_all at several imgsz on the CLEAR
out-of-sample manually-counted frames (never in training), reporting count-ratio by
manual-density bin.

Ground truth = manual box counts (frame_manifest worm_count). Frames = the already-
extracted Monday PNGs. conf 0.40 (the calibrated clear operating point).
"""

from __future__ import annotations

import csv
import glob
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
MONDAY = REPO / "validation/monday_manual_series"
MODEL = REPO / "99_runs/scaleworm_v5_all/weights/best.pt"
CONF = 0.40
BLUR_ONSET = pd.Timestamp("2023-08-11")
IMGSZS = [1280, 1536, 1920]
BINS = [(0, 8), (8, 15), (15, 22), (22, 999)]


def training_stems() -> set[str]:
    stems = set()
    for ds in ("scaleworm_retrain_all", "scaleworm_retrain_clear"):
        for p in glob.glob(str(REPO / f"datasets/{ds}/labels/train/*.txt")):
            stems.add(Path(p).stem)
    return stems


def main() -> None:
    from ultralytics import YOLO

    train = training_stems()
    man = {
        r["frame_id"]: r
        for r in csv.DictReader((MONDAY / "frame_manifest.csv").open())
        if (r.get("worm_count") or "").strip() not in ("", "nan")
    }
    # CLEAR, out-of-sample, image present
    frames = []
    for fid, r in man.items():
        if fid in train:
            continue
        if pd.to_datetime(r["datetime_utc"]) >= BLUR_ONSET:
            continue
        png = MONDAY / "images" / f"{fid}.png"
        if png.exists():
            frames.append((fid, int(float(r["worm_count"])), png))
    print(f"clear out-of-sample frames: {len(frames)}  (conf={CONF})")

    model = YOLO(str(MODEL))
    for imgsz in IMGSZS:
        by_bin: dict[tuple, list[tuple[int, int]]] = defaultdict(list)
        for fid, human, png in frames:
            res = model(str(png), conf=CONF, imgsz=imgsz, verbose=False)[0]
            n = len(res.boxes)
            for lo, hi in BINS:
                if lo <= human < hi:
                    by_bin[(lo, hi)].append((n, human))
                    break
        allpairs = [p for v in by_bin.values() for p in v]
        overall = sum(m for m, _ in allpairs) / sum(h for _, h in allpairs)
        print(f"\n=== imgsz {imgsz} === overall clear count-ratio {overall:.1%}")
        for lo, hi in BINS:
            pairs = by_bin[(lo, hi)]
            if not pairs:
                continue
            ratio = sum(m for m, _ in pairs) / sum(h for _, h in pairs)
            print(f"  density {lo:>2}-{hi:<3} n={len(pairs):3d}  "
                  f"manual/fr={sum(h for _, h in pairs) / len(pairs):5.1f}  "
                  f"count-ratio={ratio:5.1%}")


if __name__ == "__main__":
    main()

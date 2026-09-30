"""Scene-sort-free worm counting: scan the pan window, take the detector's max-count frame.

Scene-1 time varies too much recording-to-recording (median 420s, range 120-780) for a
fixed-offset extraction. Instead, sample frames across the pan window, run the detector on
each, and take the frame with the MOST worms as the recording's count -- the front-on vent
view is where the detector fires most, so max-count implicitly finds it. This needs NO
manual scene-sort, so it scales to M-F x 2015-2024.

For validation, if a row carries a manual `scene1_time_s` and/or `worm_count`, the script
also records the count at the manual time and the |best_time - manual_time| gap.

Outputs one row per recording: best_time, max_count[, count_at_manual, manual_count].

Usage:
  MODEL=99_runs/scaleworm_v5_all/weights/best.pt CONF=0.40 \
  T0=120 T1=600 STEP=60 \
  /home/jovyan/joseph-scaleworm-thesis/.venv/bin/python3 scripts/scan_count.py \
    --manifest <in.csv> --out <out.csv> [--limit N] [--stride K]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from count_frames import count_worms, extract_frame

MODEL = os.environ.get("MODEL", str(REPO / "99_runs/scaleworm_v5_all/weights/best.pt"))
CONF = float(os.environ.get("CONF", "0.40"))
T0 = float(os.environ.get("T0", "120"))
T1 = float(os.environ.get("T1", "600"))
STEP = float(os.environ.get("STEP", "60"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="cap rows (0=all)")
    ap.add_argument(
        "--stride", type=int, default=1, help="take every Kth row (even sample)"
    )
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(MODEL)
    names = model.names
    rows = list(csv.DictReader(args.manifest.open()))
    rows = rows[:: args.stride]
    if args.limit:
        rows = rows[: args.limit]
    times = [T0 + i * STEP for i in range(int((T1 - T0) / STEP) + 1)]

    import statistics

    out_rows = []
    tmp = Path(tempfile.mkdtemp())
    for k, r in enumerate(rows):
        stem = r.get("frame_id") or r.get("stem")
        video = r.get("video_path")
        if not video or not Path(video).exists():
            continue
        cnt_at_manual = ""
        man_t = (r.get("scene1_time_s") or "").strip()
        tc = []  # (time, count) for every extracted frame
        for t in times:
            png = tmp / "f.png"
            if not extract_frame(Path(video), t, png):
                continue
            res = model(str(png), conf=CONF, verbose=False)[0]
            c = count_worms([int(x) for x in res.boxes.cls.tolist()], names)
            tc.append((t, c))
            if man_t and abs(t - float(man_t)) < STEP / 2:
                cnt_at_manual = c
        if not tc:
            continue
        top = sorted(tc, key=lambda x: -x[1])[:3]  # the 3 richest frames = vent view
        med_top3 = int(statistics.median([c for _, c in top]))
        # representative time = the top-3 member whose count is the median
        best_time = min(top, key=lambda x: abs(x[1] - med_top3))[0]
        max_t, max_c = top[0]
        out_rows.append(
            {
                "frame_id": stem,
                "datetime_utc": r.get("datetime_utc", ""),
                "best_time_s": best_time,  # med-top3 representative time
                "count": med_top3,  # PRIMARY count = median of top-3 (bias-reduced)
                "max_count": max_c,  # for comparison
                "max_time_s": max_t,
                "all_counts": ";".join(
                    str(c) for _, c in tc
                ),  # raw vector, re-aggregatable
                "count_at_manual_time": cnt_at_manual,
                "manual_scene1_time_s": man_t,
                "manual_count": (r.get("worm_count") or "").strip(),
            }
        )
        if (k + 1) % 20 == 0:
            print(f"  {k + 1}/{len(rows)} scanned")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(
        f"wrote {args.out} ({len(out_rows)} recordings, window {T0:.0f}-{T1:.0f}s/{STEP:.0f}, conf {CONF})"
    )


if __name__ == "__main__":
    main()

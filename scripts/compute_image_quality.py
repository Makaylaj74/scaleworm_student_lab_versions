"""Image-quality time series: brightness, RMS contrast, and sharpness for one frame
per scene1 recording across the full 2023-2024 record.

Tests whether the optical degradation that breaks the detector (contrast -34%,
sharpness -48% between Jan-2023 and Feb-2024) is a STEP (discrete event) or a
GRADUAL slide (biofouling). Results cached per-row in a CSV so the run is resumable.
No YOLO — just ffmpeg frame extraction + opencv stats.
"""

from __future__ import annotations

import csv
import sys
import tempfile
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))
from count_frames import extract_frame, parse_stem_dt, stem_to_video

REPO = Path(__file__).resolve().parent.parent
SORT_LOG = REPO / "scene_sorting/full_2023_2024/sort_log.csv"
VIDEO_ROOT = Path("/home/jovyan/ooi/san_data/RS03ASHS-PN03B-06-CAMHDA301")
OUT_CSV = REPO / "notebooks/image_quality_timeseries.csv"
FIELDS = ["stem", "dt", "brightness", "contrast", "sharpness"]

scenes = []
with SORT_LOG.open() as fh:
    for row in csv.DictReader(fh):
        if row["decision"] == "scene1":
            scenes.append((row["stem"], float(row["scene1_time_s"])))
scenes.sort(key=lambda s: s[0])

done = set()
if OUT_CSV.exists():
    with OUT_CSV.open() as fh:
        done = {r["stem"] for r in csv.DictReader(fh)}
else:
    with OUT_CSV.open("w", newline="") as fh:
        csv.writer(fh).writerow(FIELDS)

print(f"scene1 recordings: {len(scenes)}  (already done: {len(done)})", flush=True)
t0 = time.time()
n = 0
with OUT_CSV.open("a", newline="") as fh:
    w = csv.writer(fh)
    for stem, t_s in scenes:
        if stem in done:
            continue
        vp = stem_to_video(stem, VIDEO_ROOT)
        with tempfile.TemporaryDirectory() as tmp:
            frame = Path(tmp) / f"{stem}.png"
            if vp.exists() and extract_frame(vp, t_s, frame):
                gray = cv2.cvtColor(cv2.imread(str(frame)), cv2.COLOR_BGR2GRAY)
                w.writerow(
                    [
                        stem,
                        parse_stem_dt(stem).isoformat(),
                        f"{gray.mean():.2f}",
                        f"{gray.std():.2f}",
                        f"{cv2.Laplacian(gray, cv2.CV_64F).var():.1f}",
                    ]
                )
            else:
                w.writerow([stem, parse_stem_dt(stem).isoformat(), "", "", ""])
        n += 1
        if n % 20 == 0:
            fh.flush()
            print(f"  {n} new  ({(time.time() - t0) / n:.1f}s/frame)", flush=True)
print(
    f"DONE {n} new rows in {(time.time() - t0) / 60:.1f} min -> {OUT_CSV}", flush=True
)

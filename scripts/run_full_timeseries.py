"""Run mushroom.pt (v0) over every scene1 recording in the full 2023-2024 sort.

One full-res frame per recording at its ``scene1_time_s`` -> YOLO -> count of
``scale_worm`` boxes -> one count per scene. Results cached per scene as JSON so
the run is resumable. This is the relative lower-bound abundance index feeding the
worms-over-time figure (mushroom.pt recovers ~5% of hand counts; see memory).
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from count_frames import count_worms, extract_frame, parse_stem_dt, stem_to_video
from ultralytics import YOLO

REPO = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO / "mushroom.pt"
SORT_LOG = REPO / "scene_sorting/full_2023_2024/sort_log.csv"
VIDEO_ROOT = Path("/home/jovyan/ooi/san_data/RS03ASHS-PN03B-06-CAMHDA301")
CONF_THRESHOLD = 0.25
RESULTS_DIR = REPO / "notebooks/timeseries_results_singleframe"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

scenes = []
with SORT_LOG.open() as fh:
    for row in csv.DictReader(fh):
        if row["decision"] != "scene1":
            continue
        scenes.append((row["stem"], float(row["scene1_time_s"])))
scenes.sort(key=lambda s: s[0])
print(f"scene1 recordings: {len(scenes)}", flush=True)

model = YOLO(str(MODEL_PATH))
t0 = time.time()
done = 0
for i, (stem, t_s) in enumerate(scenes):
    out = RESULTS_DIR / f"{stem}.json"
    if out.exists():
        done += 1
        continue
    vp = stem_to_video(stem, VIDEO_ROOT)
    rec = {
        "stem": stem,
        "dt": parse_stem_dt(stem).isoformat(),
        "scene1_time_s": t_s,
        "count": None,
        "video": str(vp),
    }
    if not vp.exists():
        rec["error"] = "video_missing"
    else:
        with tempfile.TemporaryDirectory() as tmp:
            frame = Path(tmp) / f"{stem}.png"
            if extract_frame(vp, t_s, frame):
                preds = model.predict(str(frame), conf=CONF_THRESHOLD, verbose=False)
                rec["count"] = count_worms(
                    [int(b.cls) for b in preds[0].boxes], model.names
                )
            else:
                rec["error"] = "frame_extract_failed"
    out.write_text(json.dumps(rec))
    done += 1
    if done % 20 == 0 or i == len(scenes) - 1:
        rate = (time.time() - t0) / max(1, done)
        print(f"  {done}/{len(scenes)}  ({rate:.1f}s/scene)", flush=True)

print(f"DONE {done}/{len(scenes)} in {(time.time() - t0) / 60:.1f} min", flush=True)

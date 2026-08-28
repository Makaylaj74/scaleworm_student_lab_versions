"""Set up a hand-count validation month: extract one frame per scene1 recording
and write a blank hand-count CSV. Mirrors the Feb-2024 setup so a second month can
be counted the same way (validates whether a given regime's model counts are real).

Usage:  python scripts/setup_handcount_month.py 2023/01
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from count_frames import extract_frame, parse_stem_dt, stem_to_video

REPO = Path(__file__).resolve().parent.parent
SORT_LOG = REPO / "scene_sorting/full_2023_2024/sort_log.csv"
VIDEO_ROOT = Path("/home/jovyan/ooi/san_data/RS03ASHS-PN03B-06-CAMHDA301")

month = sys.argv[1] if len(sys.argv) > 1 else "2023/01"
tag = month.replace("/", "_")
month_stem = "-" + month.replace("/", "")
FRAMES_DIR = REPO / f"notebooks/handcount_{tag}_frames"
HANDCOUNT_CSV = REPO / f"notebooks/handcount_{tag}.csv"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

scenes = []
with SORT_LOG.open() as fh:
    for row in csv.DictReader(fh):
        if row["decision"] == "scene1" and month_stem in row["stem"]:
            scenes.append((row["stem"], float(row["scene1_time_s"])))
scenes.sort(key=lambda s: s[0])
print(f"{month}: {len(scenes)} scene1 recordings")

for stem, t_s in scenes:
    out = FRAMES_DIR / f"{stem}.png"
    if out.exists():
        continue
    vp = stem_to_video(stem, VIDEO_ROOT)
    if not (vp.exists() and extract_frame(vp, t_s, out)):
        print(f"  FAILED extract {stem} (video_exists={vp.exists()})")
print(f"frames on disk: {len(list(FRAMES_DIR.glob('*.png')))}")

# blank hand-count template (do not clobber counts already entered)
existing = {}
if HANDCOUNT_CSV.exists():
    with HANDCOUNT_CSV.open() as fh:
        existing = {r["stem"]: r.get("human_count", "") for r in csv.DictReader(fh)}
with HANDCOUNT_CSV.open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["stem", "dt", "scene1_time_s", "human_count"])
    for stem, t_s in scenes:
        w.writerow([stem, parse_stem_dt(stem).isoformat(), t_s, existing.get(stem, "")])
print(
    f"wrote {HANDCOUNT_CSV}  ({sum(1 for v in existing.values() if str(v).strip())} filled)"
)

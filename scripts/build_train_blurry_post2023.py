"""Build the Phase-2 (blurry post-2023) training manifest for the v3 retrain.

Unlike the -2021 batch, no scene-picking is needed: the full 2023-2024 sort already
supplies scene1_time_s for every post-Aug-2023 recording. This selects an even-spread
sample across the blurry period, EXCLUDING the Feb-2024 + May-2024 hand-count months
(they are the held-out acceptance gate) so the retrain stays leakage-free.

Writes datasets/scaleworm_v2/train_frames_blurry_post2023.csv (stem, scene1_time_s,
video_path) -- the same columns make_prelabels.py consumes.

    python scripts/build_train_blurry_post2023.py            # N=40 default
    N=60 python scripts/build_train_blurry_post2023.py
"""

from __future__ import annotations

import csv
import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path("/home/jovyan/scaleworm-student-lab")
sys.path.insert(0, str(REPO / "scripts"))
from count_frames import stem_to_video  # noqa: E402

SORT_LOG = REPO / "scene_sorting/full_2023_2024/sort_log.csv"
VIDEO_ROOT = Path("/home/jovyan/ooi/san_data/RS03ASHS-PN03B-06-CAMHDA301")
OUT = REPO / "datasets/scaleworm_v2/train_frames_blurry_post2023.csv"
BLUR_ONSET = datetime(2023, 8, 1)
GATE_MONTHS = {(2024, 2), (2024, 5)}  # Feb + May 2024 = held-out hand-count gate
N = int(os.environ.get("N", "40"))


def stem_dt(stem: str) -> datetime:
    return datetime.strptime(
        re.search(r"(\d{8}T\d{6})", stem).group(1), "%Y%m%dT%H%M%S"
    )


def main() -> None:
    scene1 = [
        r
        for r in csv.DictReader(SORT_LOG.open())
        if r["decision"] == "scene1" and (r.get("scene1_time_s") or "").strip()
    ]
    pool = []
    for r in scene1:
        d = stem_dt(r["stem"])
        if d >= BLUR_ONSET and (d.year, d.month) not in GATE_MONTHS:
            pool.append((r["stem"], r["scene1_time_s"], d))
    pool.sort(key=lambda x: x[2])
    if len(pool) <= N:
        picks = pool
    else:
        idx = np.linspace(0, len(pool) - 1, N).round().astype(int)
        picks = [pool[i] for i in sorted(set(idx))]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    n_missing = 0
    with OUT.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stem", "scene1_time_s", "video_path"])
        for stem, t, _ in picks:
            vp = stem_to_video(stem, VIDEO_ROOT)
            if not vp.exists():
                n_missing += 1
                continue
            w.writerow([stem, t, str(vp)])

    mc = Counter((d.year, d.month) for _, _, d in picks)
    print(f"pool (post-Aug-2023, excl. Feb/May-2024 gate): {len(pool)}")
    print(
        f"selected {len(picks)} even-spread ({n_missing} dropped: video missing) -> {OUT}"
    )
    for k in sorted(mc):
        print(f"  {k[0]}-{k[1]:02d}: {mc[k]}")


if __name__ == "__main__":
    main()

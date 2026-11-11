"""Automated worm index, Jan 2021 - Aug 2023, read at each camera-schedule era's Scene-1 time.

Replaces the max-over-grid frame choice of build_worm_series_2021_2023_density.py, which
under-read Jan-Aug 2021 (0.38-0.51x hand counts). Validation (scripts/era_finder_score.py,
method_writeup_detector §5): on 91 rule-counted 2021 Monday frames the era-time read gives
r = 0.80 and a period-consistent ratio (0.82 / 0.86); on independent Tuesdays it improves
r over the old frame choice (per frame 0.56 -> 0.64).

Per recording (the 5354 clear Mon-Fri recordings of the v8+SAHI harvest lists):
  1. read ONE frame at the era's Scene-1 time (ERA rules, derived from the Monday sort only);
     where an era has two candidate times, read both and keep the higher
  2. density head (csrnet_run/best.pt) count / 0.96, same scale as the previous index
Aggregation (aggregate()):
  3. drop recordings reading < OFFSCENE_CUTOFF (8.5, chosen on Monday Jan-Aug 2021 sort
     labels only; Tuesday transfer sens 0.90 / spec 0.83) as off-scene
  4. day mean of kept recordings -> week (W-MON) mean over days, SEM over days

Resumable and shardable (24-worker lab cap: 3 shards x OMP/torch 8 threads):
    for i in 0 1 2; do OMP_NUM_THREADS=8 SHARD=$i NSHARDS=3 \\
        .venv/bin/python3 scripts/build_era_time_index_2021_2023.py read & done
    .venv/bin/python3 scripts/build_era_time_index_2021_2023.py aggregate
"""

from __future__ import annotations

import csv
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "notebooks/era_time_index_2021_2023"
WEEKLY = OUT / "era_time_index_weekly_2021_2023.csv"
OFFSCENE_CUTOFF = 8.5
# (first YYYYMMDD inclusive, Scene-1 time(s) in s); from the Monday sort, first match wins
ERA = [
    ("20230601", (210,)),
    ("20230101", (210, 420)),
    ("20220425", (210,)),
    ("20210901", (540,)),
    ("00000000", (420,)),
]


def era_times(day: str) -> tuple[int, ...]:
    """Scene-1 time(s) for a YYYYMMDD day."""
    for start, times in ERA:
        if day >= start:
            return times
    raise ValueError(day)


def aggregate(reads: pd.DataFrame, cutoff: float = OFFSCENE_CUTOFF) -> pd.DataFrame:
    """Weekly index from per-recording reads (columns frame_id, worm_index)."""
    r = reads[reads.worm_index >= cutoff].copy()
    r["day"] = pd.to_datetime(r.frame_id.str[10:18])
    day = r.groupby("day").agg(
        day_mean=("worm_index", "mean"), n_rec=("worm_index", "size")
    )
    day["week"] = day.index.to_period("W-SUN").start_time
    g = day.groupby("week")
    wk = pd.DataFrame(
        {
            "worm_index": g.day_mean.mean(),
            "sem": g.day_mean.std() / np.sqrt(g.day_mean.size()),
            "n_days": g.day_mean.size(),
            "n_recordings": g.n_rec.sum(),
        }
    )
    total = reads.assign(
        week=pd.to_datetime(reads.frame_id.str[10:18])
        .dt.to_period("W-SUN")
        .dt.start_time
    )
    wk["n_offscene"] = total.groupby("week").size().reindex(wk.index) - wk.n_recordings
    return wk


def read(shard: int, nshards: int) -> None:
    import cv2
    import torch
    from PIL import Image
    from torchvision import transforms

    sys.path.insert(0, str(REPO / "notebooks"))
    import build_worm_series_2021_2023_density as B
    import density_head as D

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    recs = B.load_harvest()[shard::nshards]
    out = OUT / f"reads_shard{shard}.csv"
    done = set(pd.read_csv(out).frame_id) if out.exists() else set()
    model = D.CSRNet()
    model.load_state_dict(torch.load(B.CLEAR_MODEL, map_location="cpu"))
    model.eval()
    norm = transforms.Normalize(D.IMAGENET_MEAN, D.IMAGENET_STD)
    print(
        f"shard {shard}/{nshards}: {len(recs)} recordings, {len(done)} done", flush=True
    )
    new = not out.exists()
    with out.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["frame_id", "t_s", "worm_index", "n_candidates"])
        for i, rec in enumerate(recs):
            fid = rec["frame_id"]
            if fid in done:
                continue
            cap = cv2.VideoCapture(str(B.video_path(fid, rec["dt"].astimezone(UTC))))
            best = None
            times = era_times(fid[10:18])
            for t in times:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ok, fr = cap.read()
                if ok:
                    img = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                    c = B.density_count(model, img, "cpu", norm) / B.RATIO
                    if best is None or c > best[1]:
                        best = (t, c)
            cap.release()
            if best is not None:
                w.writerow([fid, best[0], round(best[1], 3), len(times)])
            if i % 100 == 0:
                fh.flush()
                print(
                    f"  shard {shard}: {i + 1}/{len(recs)} {datetime.now(UTC):%H:%M}",
                    flush=True,
                )
    print(f"shard {shard} done", flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1:] == ["read"]:
        read(int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1")))
    elif sys.argv[1:] == ["aggregate"]:
        reads = pd.concat(pd.read_csv(p) for p in sorted(OUT.glob("reads_shard*.csv")))
        reads = reads.drop_duplicates("frame_id")
        wk = aggregate(reads)
        wk.to_csv(WEEKLY, index_label="week")
        print(f"{len(reads)} recordings -> {len(wk)} weeks -> {WEEKLY}")
    else:
        raise SystemExit("usage: build_era_time_index_2021_2023.py read|aggregate")


if __name__ == "__main__":
    main()

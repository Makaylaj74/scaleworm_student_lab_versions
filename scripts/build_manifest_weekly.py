"""Build the weekly-cadence sampling manifest for the CLEAR window (CAMHDA301
-2021 & -2022 units): 2021-08-01 .. 2023-08-01, ending just before the soft-focus
Aug-2023 camera swap.

Pipeline: build_manifest_weekly.py -> compute_image_quality_multiyear.py <manifest>
<out> -> aggregate_image_quality_weekly.py.

Deterministic (no RNG): per ISO week, take all REAL .mp4 sorted by timestamp and
pick up to N evenly-spaced (spreads over days-of-week and hours-UTC).
"""

from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path("/home/jovyan/ooi/san_data/RS03ASHS-PN03B-06-CAMHDA301")
# naive UTC on purpose: filename timestamps are UTC ('...Z'), compared consistently
START, END = datetime(2021, 8, 1), datetime(2023, 8, 1)  # noqa: DTZ001
PER_WEEK = 8
OUT = Path("/tmp/camhd_survey/manifest_weekly.csv")
pat = re.compile(r"CAMHDA301-(\d{8}T\d{6})")


def main() -> None:
    buckets: dict[tuple[int, int], list[tuple[str, str]]] = {}
    for ydir in (ROOT / "2021", ROOT / "2022", ROOT / "2023"):
        for f in ydir.rglob("*.mp4"):
            if not f.exists():  # skip dead symlinks
                continue
            m = pat.search(f.name)
            if not m:
                continue
            dt = datetime.strptime(m.group(1), "%Y%m%dT%H%M%S")  # noqa: DTZ007
            if not (START <= dt < END):
                continue
            iso = dt.isocalendar()
            buckets.setdefault((iso.year, iso.week), []).append((m.group(1), str(f)))

    rows = []
    for key in sorted(buckets):
        files = sorted(buckets[key])
        if len(files) <= PER_WEEK:
            pick = files
        else:
            idx = sorted(
                set(np.linspace(0, len(files) - 1, PER_WEEK).astype(int).tolist())
            )
            pick = [files[i] for i in idx]
        for ts, path in pick:
            rows.append((f"{key[0]}-W{key[1]:02d}", ts, path))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ym", "ts", "path"])  # 'ym' holds ISO-week label (reuses schema)
        w.writerows(rows)

    print(f"sampled {len(rows)} recordings across {len(buckets)} weeks -> {OUT}")


if __name__ == "__main__":
    main()

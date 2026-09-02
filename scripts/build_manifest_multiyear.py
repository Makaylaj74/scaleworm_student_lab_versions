"""Build the stratified sampling manifest for the multi-year CAMHDA301 image-quality
survey (2015-2026).

Pipeline: build_manifest_multiyear.py -> compute_image_quality_multiyear.py
<manifest> <out> -> aggregate_image_quality_multiyear.py.

Deterministic (no RNG): per year-month, take all REAL .mp4 recordings sorted by
timestamp and pick N evenly-spaced across the sorted list (spreads over
days-of-month and hours-UTC). The paired .mov files are dead symlinks to an
unmounted /MOVs volume, so only .mp4 that actually resolve are used.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

ROOT = Path("/home/jovyan/ooi/san_data/RS03ASHS-PN03B-06-CAMHDA301")
N_PER_MONTH = 12
OUT = Path("/tmp/camhd_survey/manifest_multiyear.csv")
pat = re.compile(r"CAMHDA301-(\d{8}T\d{6})")


def main() -> None:
    rows = []
    for ydir in sorted(ROOT.glob("20[0-9][0-9]")):
        for mdir in sorted(ydir.glob("[01][0-9]")):
            files = []
            for f in mdir.rglob("*.mp4"):
                if not f.exists():  # skip dead symlinks
                    continue
                m = pat.search(f.name)
                if m:
                    files.append((m.group(1), str(f)))
            if not files:
                continue
            files.sort()
            if len(files) <= N_PER_MONTH:
                pick = files
            else:
                idx = sorted(
                    set(
                        np.linspace(0, len(files) - 1, N_PER_MONTH).astype(int).tolist()
                    )
                )
                pick = [files[i] for i in idx]
            for ts, path in pick:
                rows.append((f"{ydir.name}-{mdir.name}", ts, path))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ym", "ts", "path"])
        w.writerows(rows)

    months = len({r[0] for r in rows})
    print(f"sampled {len(rows)} recordings across {months} months -> {OUT}")


if __name__ == "__main__":
    main()

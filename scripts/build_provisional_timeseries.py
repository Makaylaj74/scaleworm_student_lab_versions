"""Provisional weekly worm-abundance series for the geophysics overlay.

2023-2024 Mondays use MANUAL box-corrected counts (frame_status=counted,
worm_count from the manifest = ground truth). 2021-2022 Mondays are PROVISIONAL:
worm_count = number of v2 pre-label boxes (len of the label file), because those
frames are not yet human-corrected. v2 clear recall is ~66-90% so 2021-2022
points are a LOWER BOUND and are NOT level-comparable to the 2023-2024 manual
points until box-corrected — the output carries a `method` column so the plot
can distinguish them.

Output: notebooks/worm_timeseries_provisional_2017_2024.csv
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "validation/monday_manual_series/frame_manifest.csv"
MANIFEST_DIR = MANIFEST.parent
OUT = REPO / "notebooks/worm_timeseries_provisional_2017_2024.csv"


def _count_label(label_path: str) -> int | None:
    # label_path in the manifest is relative to the manifest's own directory
    if not isinstance(label_path, str) or not label_path:
        return None
    p = Path(label_path)
    if not p.is_absolute():
        p = MANIFEST_DIR / label_path
    if not p.exists():
        return None
    return sum(1 for ln in p.read_text().splitlines() if ln.strip())


def load_records() -> list[tuple[date, int, str]]:
    df = pd.read_csv(MANIFEST)
    recs: list[tuple[date, int, str]] = []
    for _, r in df.iterrows():
        d = datetime.fromisoformat(r["datetime_utc"]).date()
        status = str(r.get("frame_status", ""))
        if status == "counted" and pd.notna(r.get("worm_count")):
            recs.append((d, int(r["worm_count"]), "manual"))
        elif d.year in (2021, 2022):
            n = _count_label(r.get("label_path"))
            if n is not None:
                recs.append((d, n, "v2_provisional"))
    return recs


def aggregate(records: list[tuple[date, int, str]]) -> list[dict]:
    by_day: dict[date, list[int]] = defaultdict(list)
    method: dict[date, str] = {}
    for d, c, m in records:
        by_day[d].append(c)
        method[d] = m
    rows = []
    for d in sorted(by_day):
        counts = by_day[d]
        n = len(counts)
        mean = statistics.fmean(counts)
        if n > 1:
            std = statistics.stdev(counts)
            sem = std / math.sqrt(n)
        else:
            std = sem = float("nan")
        rows.append(
            {
                "date": d.isoformat(),
                "n_slots": n,
                "total_worms": sum(counts),
                "mean_worms": round(mean, 4),
                "std_worms": round(std, 4) if not math.isnan(std) else "",
                "sem_worms": round(sem, 4) if not math.isnan(sem) else "",
                "method": method[d],
            }
        )
    return rows


def main() -> None:
    recs = load_records()
    rows = aggregate(recs)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    nm = sum(r["method"] == "manual" for r in rows)
    nv = sum(r["method"] == "v2_provisional" for r in rows)
    print(
        f"{len(recs)} frames -> {len(rows)} Mondays "
        f"({nm} manual, {nv} v2-provisional). wrote {OUT}"
    )


if __name__ == "__main__":
    main()

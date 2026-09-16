"""Aggregate the Monday manual Scene-1 series into a per-Monday abundance timeseries.

Reads ``validation/monday_manual_series/frame_manifest.csv`` (one row per Scene-1
frame, ``worm_count`` derived from the corrected boxes) and rolls the up-to-eight
three-hourly slots of each Monday into a single point estimate with an error bar:

    point   = mean worm_count across that Monday's counted slots
    spread  = SEM = sample-std / sqrt(n)   (NaN when only one slot was counted)

Only ``frame_status == counted`` rows enter the series. ``no_scene1`` / ``unusable_blur``
rows are missing data, not observed zeros (schema rule 2), so they are excluded — a
Monday with no counted slot simply has no point.

Writes ``validation/monday_manual_series/monday_manual_timeseries.csv``.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

REPO = Path("/home/jovyan/scaleworm-student-lab")
DEFAULT_MANIFEST = REPO / "validation/monday_manual_series/frame_manifest.csv"


def aggregate_by_monday(
    records: list[tuple[date, int]],
) -> list[dict]:
    """Roll (recording_date, worm_count) pairs into per-day mean/SEM rows.

    Args:
        records: one ``(date, worm_count)`` per counted Scene-1 frame.

    Returns:
        One dict per distinct date, sorted ascending, with keys
        ``date, n_slots, total_worms, mean_worms, std_worms, sem_worms``.
        ``std``/``sem`` are ``float('nan')`` when ``n_slots == 1``.
    """
    by_day: dict[date, list[int]] = defaultdict(list)
    for d, count in records:
        by_day[d].append(count)

    out = []
    for d in sorted(by_day):
        counts = by_day[d]
        n = len(counts)
        mean = statistics.fmean(counts)
        if n >= 2:
            std = statistics.stdev(counts)  # sample std, ddof=1
            sem = std / math.sqrt(n)
        else:
            std = sem = float("nan")
        out.append(
            {
                "date": d.isoformat(),
                "n_slots": n,
                "total_worms": sum(counts),
                "mean_worms": round(mean, 4),
                "std_worms": round(std, 4) if not math.isnan(std) else "",
                "sem_worms": round(sem, 4) if not math.isnan(sem) else "",
            }
        )
    return out


def _parse_count(value: str) -> int:
    """Parse a ``worm_count`` cell to an int.

    The manifest carries mixed string formats: nb 33's labeler writes plain ints
    (``"8"``), while the earlier 2023-2024 pipeline wrote float strings (``"31.0"``).
    All values are whole numbers, so ``int(float(...))`` normalises both without loss.
    """
    return int(float(value))


def _load_counted(manifest: Path) -> list[tuple[date, int]]:
    recs = []
    with manifest.open(newline="") as fh:
        for r in csv.DictReader(fh):
            if r["frame_status"] != "counted":
                continue
            d = datetime.fromisoformat(r["datetime_utc"]).date()
            recs.append((d, _parse_count(r["worm_count"])))
    return recs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest", type=Path, nargs="?", default=DEFAULT_MANIFEST)
    args = ap.parse_args()

    recs = _load_counted(args.manifest)
    series = aggregate_by_monday(recs)

    out = args.manifest.parent / "monday_manual_timeseries.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "date",
                "n_slots",
                "total_worms",
                "mean_worms",
                "std_worms",
                "sem_worms",
            ],
        )
        w.writeheader()
        w.writerows(series)

    n_pts = len(series)
    n_err = sum(1 for s in series if s["sem_worms"] != "")
    print(f"{len(recs)} counted frames -> {n_pts} Mondays ({n_err} with a SEM).")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

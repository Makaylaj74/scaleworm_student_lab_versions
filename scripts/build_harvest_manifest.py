"""Enumerate CamHD recordings for the M-F harvest over a date range.

Walks the on-disk archive (YYYY/MM/DD/*.mp4), keeps recordings on the requested weekdays,
and writes a manifest (frame_id, datetime_utc, video_path) that harvest_series.py consumes.

Usage:
  python scripts/build_harvest_manifest.py --start 2023-03-01 --end 2023-03-31 \
     --weekdays 1-4 --out <manifest.csv>     # weekdays: 0=Mon..6=Sun; "1-4"=Tue-Fri
"""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

ARCHIVE = Path("/home/jovyan/ooi/san_data/RS03ASHS-PN03B-06-CAMHDA301")
# The 8 canonical daily recording slots (every 3 h). 2017+ record at HH:15:00, 2015-2016 at
# HH:00:00 — accept both families so one filter covers all eras (no day uses both). Some days
# carry extra non-standard/high-rate recordings; the goal is a consistent 8 frames/day.
CANONICAL_SLOTS = frozenset(
    {f"{h:02d}{m}" for h in (0, 3, 6, 9, 12, 15, 18, 21) for m in ("1500", "0000")}
)


def parse_weekdays(spec: str) -> set[int]:
    out: set[int] = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def dt_from_stem(stem: str) -> datetime:
    ts = stem.split("-")[1][:15]  # YYYYMMDDTHHMMSS (CamHD timestamps are UTC)
    return datetime.strptime(ts, "%Y%m%dT%H%M%S").replace(tzinfo=UTC)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--weekdays", default="0-4", help="0=Mon..6=Sun; e.g. 1-4=Tue-Fri")
    ap.add_argument(
        "--all-slots",
        action="store_true",
        help="keep every recording (default: only the 8 canonical slots)",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    wds = parse_weekdays(args.weekdays)
    d0 = date.fromisoformat(args.start)
    d1 = date.fromisoformat(args.end)
    rows = []
    d = d0
    while d <= d1:
        if d.weekday() in wds:
            day_dir = ARCHIVE / f"{d.year}" / f"{d.month:02d}" / f"{d.day:02d}"
            for mp4 in sorted(day_dir.glob("*.mp4")):
                stem = mp4.stem
                try:
                    dt = dt_from_stem(stem)
                except (ValueError, IndexError):
                    continue
                if (
                    not args.all_slots
                    and stem.split("-")[1][9:15] not in CANONICAL_SLOTS
                ):
                    continue
                rows.append(
                    {
                        "frame_id": stem,
                        "datetime_utc": dt.isoformat(),
                        "video_path": str(mp4),
                    }
                )
        d += timedelta(days=1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame_id", "datetime_utc", "video_path"])
        w.writeheader()
        w.writerows(rows)
    from collections import Counter

    by_wd = Counter(date.fromisoformat(r["datetime_utc"][:10]).weekday() for r in rows)
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    print(f"{len(rows)} recordings {args.start}..{args.end} weekdays {sorted(wds)}")
    print("  by weekday:", {names[k]: v for k, v in sorted(by_wd.items())})
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

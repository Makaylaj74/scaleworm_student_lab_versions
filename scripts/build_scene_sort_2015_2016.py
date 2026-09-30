"""Build the 2015-2016 weekly scene-sort set (cadence-agnostic).

The 2015-2016 CAMHD archive does NOT follow the fixed 3-hourly :15 slot schedule
that ``scene_sampler.py`` assumes for 2021-2024: recordings carry a ``Z`` suffix,
sit at :00:00 (late-2015..2016) or irregular times (early Jul-2015), and late-2016
switches to paired :xx:07/:xx:14 clips. So we can't enumerate fixed slots. Instead,
for each sampled Monday we glob whatever recordings exist that day, keep normal
~14-min pans, and pick up to 8 evenly across the day to mirror the 2021-2024
"up to 8 three-hourly slots per Monday" aggregation.

Output mirrors the 2021-2024 sort layout so the existing sorter + build_frame_manifest
apply: ``<out>/{contact_sheets,scene1,not_scene1}`` + ``manifest.csv``.

    python scripts/build_scene_sort_2015_2016.py --start 2015-07-13 --end 2015-12-28 \
        --out scene_sorting/full_2015
"""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from scene_sampler import build_contact_sheet, iter_sample_dates, probe_duration

VROOT = Path("/home/jovyan/ooi/san_data/RS03ASHS-PN03B-06-CAMHDA301")
REPO = Path("/home/jovyan/scaleworm-student-lab")
DUR_LO, DUR_HI = 300.0, 1800.0  # normal ~14-min pans only
MAX_PER_DAY = 8                 # mirror the 2021-2024 8-slot-per-Monday structure
WORKERS = 8                     # concurrent sheets; <= 24-worker cap


def _dt(p: Path) -> datetime:
    stem = p.stem.split("-")[1].rstrip("Z")
    return datetime.strptime(stem[:15], "%Y%m%dT%H%M%S")


def even(items: list, k: int) -> list:
    if k >= len(items):
        return items
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def recordings_on(day: date, max_per_day: int = MAX_PER_DAY) -> list[Path]:
    """All normal-duration recordings on `day`, up to max_per_day, evenly spread."""
    d = VROOT / f"{day.year:04d}/{day.month:02d}/{day.day:02d}"
    cands = sorted(d.glob("*.mp4"), key=_dt)
    good = []
    for p in cands:
        dur = probe_duration(p)
        if dur is not None and DUR_LO <= dur <= DUR_HI:
            good.append(p)
    return even(good, max_per_day)


def available_days(start: date, end: date) -> list[date]:
    """Distinct dates in [start, end] that have at least one recording."""
    out = set()
    for p in VROOT.glob("201[56]/*/*/*.mp4"):
        y, m, d = int(p.parts[-4]), int(p.parts[-3]), int(p.parts[-2])
        dd = date(y, m, d)
        if start <= dd <= end:
            out.add(dd)
    return sorted(out)


def sample_days(start: date, end: date, pick: str, per_month: int = 2) -> list[date]:
    """Days to sample. 'monday' = fixed weekly Mondays (daily-record eras like 2016);
    'available-week' = one data-bearing day per ISO week (sporadic eras like 2015),
    choosing the day with the most recordings that week; 'monthly' = per_month
    data-bearing days evenly spread within each calendar month (a coarse recovery-
    cadence subsample — slow succession signal doesn't need weekly resolution)."""
    if pick == "monday":
        return iter_sample_dates(start, end, weekday=0, step_weeks=1)
    if pick == "monthly":
        by_month: dict[tuple, list[date]] = {}
        for d in available_days(start, end):
            by_month.setdefault((d.year, d.month), []).append(d)
        chosen = []
        for mo in sorted(by_month):
            chosen.extend(even(sorted(by_month[mo]), per_month))
        return chosen
    by_week: dict[tuple, list[date]] = {}
    for d in available_days(start, end):
        by_week.setdefault(d.isocalendar()[:2], []).append(d)
    chosen = []
    for wk in sorted(by_week):
        days = by_week[wk]
        # day with most recordings (tie -> earliest) best represents that week
        chosen.append(max(days, key=lambda d: (len(list(
            (VROOT / f"{d.year:04d}/{d.month:02d}/{d.day:02d}").glob("*.mp4"))), -d.toordinal())))
    return chosen


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=date.fromisoformat, default=date(2015, 7, 13))
    ap.add_argument("--end", type=date.fromisoformat, default=date(2015, 12, 28))
    ap.add_argument("--out", type=Path, default=REPO / "scene_sorting/full_2015")
    ap.add_argument("--pick", choices=("monday", "available-week", "monthly"),
                    default="available-week",
                    help="'available-week' for sporadic 2015; 'monday' for daily-record "
                         "2016; 'monthly' for a coarse per_month/mo recovery subsample.")
    ap.add_argument("--per-month", type=int, default=2,
                    help="days per calendar month when --pick monthly (default 2).")
    ap.add_argument("--max-per-day", type=int, default=MAX_PER_DAY,
                    help="max recordings sheeted per sampled day (default 8).")
    ap.add_argument("--workers", type=int, default=WORKERS)
    args = ap.parse_args(argv)

    out = args.out if args.out.is_absolute() else REPO / args.out
    sheets = out / "contact_sheets"
    for sub in ("contact_sheets", "scene1", "not_scene1"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    days = sample_days(args.start, args.end, args.pick, args.per_month)
    picks: list[Path] = []
    for d in days:
        recs = recordings_on(d, args.max_per_day)
        picks.extend(recs)
        print(f"  {d} ({d.strftime('%a')}): {len(recs)} recordings")
    print(f"{len(days)} sampled days ({args.pick}) -> {len(picks)} recordings to sheet ({args.workers} workers)")

    rows: list[dict] = []

    def render(vp: Path) -> dict | None:
        outp = sheets / f"{vp.stem}.png"
        if not outp.exists() and build_contact_sheet(vp, outp) is None:
            print(f"  FAIL {vp.stem}")
            return None
        dt = _dt(vp)
        return {
            "stem": vp.stem,
            "datetime_utc": dt.isoformat(),
            "recording_date": dt.date().isoformat(),
            "sheet_path": str(outp.relative_to(out)),
            "video_path": str(vp),
        }

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(render, vp): vp for vp in picks}
        for f in as_completed(futs):
            r = f.result()
            done += 1
            if r:
                rows.append(r)
            if done % 20 == 0:
                print(f"[{done}/{len(picks)}] sheets built")

    rows.sort(key=lambda r: r["stem"])
    man = out / "manifest.csv"
    with man.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {man} ({len(rows)} sheets); sort folders under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

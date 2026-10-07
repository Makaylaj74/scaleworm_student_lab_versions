"""Build the 2015 count-series expansion annotation set.

Metric decision (2026-10-07): COUNT throughout, for 2015-2024 continuity. The detector is
OOD on the 2015 dense mound (count-ratio 0.23, r=0.38) so this era stays fully manual. The
42-frame density-stratified subsample already anchored the level and covers 20/21 weeks with
one frame each; this set lets the counter ADD per-week replicates (tighter error bars) by
hand-counting the remaining Scene-1 frames.

Crunch-aware design: all remaining Scene-1 frames are extracted, but the sheet is ORDERED by
priority -- empty/sparsest weeks first, then round-robin across weeks -- so counting top-down
and stopping at any point yields an evenly distributed weekly series. You do NOT have to finish.

Run (thesis venv has the SAN mount):
  /home/jovyan/joseph-scaleworm-thesis/.venv/bin/python scripts/build_expand_2015_handcount.py
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from count_frames import extract_frame

REPO = Path("/home/jovyan/scaleworm-student-lab")
SORT = REPO / "scene_sorting" / "full_2015"
DONE_SHEET = (
    REPO / "validation" / "subsample_2015_2016_stratified" / "handcount_sheet.csv"
)
SCORES = REPO / "validation" / "subsample_2015_2016_stratified" / "detector_scores.csv"
OUT = REPO / "validation" / "expand_2015_handcount"

FIELDS = [
    "frame_id",
    "year",
    "scene1_time_s",
    "detector_count",
    "worm_count",
    "percent_cover",
    "counter",
    "metric_note",
    "notes",
    "video_path",
]


def _week(stem: str) -> tuple[int, int]:
    """(iso-year, iso-week) from CAMHDA301-20150716T000015Z."""
    d = datetime.strptime(stem.split("-")[1][:8], "%Y%m%d")
    iso = d.isocalendar()
    return (iso.year, iso.week)


def main() -> None:
    # Scene-1 pool for 2015 (stem -> scene1_time_s) and video paths.
    times = {
        r["stem"]: float(r["scene1_time_s"])
        for r in csv.DictReader((SORT / "sort_log.csv").open())
        if r["decision"] == "scene1" and r.get("scene1_time_s")
    }
    vids = {
        r["stem"]: r.get("video_path", "")
        for r in csv.DictReader((SORT / "manifest.csv").open())
    }
    scores = {}
    if SCORES.exists():
        scores = {r["stem"]: r["det_count"] for r in csv.DictReader(SCORES.open())}

    # Frames already handled in the subsample (counted or screened-blurry) -> skip.
    handled = {
        r["frame_id"] for r in csv.DictReader(DONE_SHEET.open()) if r["year"] == "2015"
    }
    counted_per_week: dict[tuple[int, int], int] = defaultdict(int)
    for r in csv.DictReader(DONE_SHEET.open()):
        if r["year"] == "2015" and r["worm_count"] != "":
            counted_per_week[_week(r["frame_id"])] += 1

    todo = sorted(s for s in times if s not in handled)

    # Group remaining frames by week; order weeks by (existing counts asc, week asc).
    by_week: dict[tuple[int, int], list[str]] = defaultdict(list)
    for s in todo:
        by_week[_week(s)].append(s)
    weeks = sorted(by_week, key=lambda w: (counted_per_week[w], w))
    for w in weeks:
        by_week[w].sort()

    # Round-robin interleave so any prefix spreads evenly across weeks.
    order: list[str] = []
    depth = 0
    while any(len(by_week[w]) > depth for w in weeks):
        for w in weeks:
            if len(by_week[w]) > depth:
                order.append(by_week[w][depth])
        depth += 1

    (OUT / "frames").mkdir(parents=True, exist_ok=True)
    (OUT / "clicks").mkdir(exist_ok=True)
    rows, extracted, missing = [], 0, 0
    for stem in order:
        vp = vids.get(stem, "")
        png = OUT / "frames" / f"{stem}.png"
        if not vp or not Path(vp).exists():
            missing += 1
            continue
        if not png.exists() and not extract_frame(Path(vp), times[stem], png):
            missing += 1
            continue
        extracted += 1
        rows.append(
            {
                "frame_id": stem,
                "year": "2015",
                "scene1_time_s": times[stem],
                "detector_count": scores.get(stem, ""),
                "worm_count": "",
                "percent_cover": "",
                "counter": "",
                "metric_note": "count (metric decided 2026-10-07)",
                "notes": "",
                "video_path": vp,
            }
        )

    with (OUT / "handcount_sheet.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    (OUT / "README.md").write_text(
        "*AI-generated draft (Claude, Anthropic) — for review. Frames and ordering come "
        "from version-controlled scripts; the worm counts you enter are your own data.*\n\n"
        "# 2015 count-series expansion (manual)\n\n"
        f"{len(rows)} Scene-1 frames (2015, Jul-Dec), the remainder after the 42-frame "
        "density-stratified subsample. Metric = **individual count** (decided 2026-10-07, for "
        "2015-2024 continuity). The detector is OOD here (ratio 0.23, r=0.38) so counts are "
        "fully manual.\n\n"
        "Rows are **priority-ordered** (empty/sparsest weeks first, then round-robin across "
        "weeks). Count top-down in notebook 29 and **stop whenever** -- any prefix gives an "
        "evenly distributed weekly series. Flags land in `notes`; `metric_note` is preserved.\n\n"
        "Frames are gitignored (regenerable from `scripts/build_expand_2015_handcount.py`).\n"
    )

    print(
        f"wrote {OUT}/handcount_sheet.csv -- {len(rows)} frames extracted "
        f"({missing} skipped/missing)"
    )
    print(f"weeks represented: {len({_week(r['frame_id']) for r in rows})}")
    if rows:
        print(f"first 3 (highest priority): {[r['frame_id'] for r in rows[:3]]}")


if __name__ == "__main__":
    main()

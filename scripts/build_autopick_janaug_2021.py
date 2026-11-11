"""Auto-pick Scene-1 frames for the Jan-Aug 2021 Monday extension (skip the manual sort).

Advisor question (2026-11-11): is the low early-2021 automated index real? The manual
Monday series starts 2021-09-06, so Jan-Aug 2021 has no systematic human truth. Rather
than sort 276 contact sheets, MJ box-corrects auto-picked frames in nb33 and uses its
"No Scene-1" button as the in-pass sort (NA, not 0).

Pick rule: 3 FIXED slots per Monday (03:15, 12:15, 21:15 -- fixed, not top-N, so the
recording choice is not density-biased), Jan 4 - Aug 30 2021 (35 Mondays, ~105 frames).
Frame time = v8+SAHI max-over-60s-grid ``best_time_s`` from the Jan-Aug 2021 harvest.

Caveat (record in any comparison): the max-count frame time may bias counts HIGH vs the
human-picked times of the Sep-2021+ series -- conservative against a "low 2021" finding.

RESULT (2026-11-11): FAILED -- MJ found none of the first 10 auto-picked frames were
Scene-1, so the set was archived to validation/monday_janaug_2021_autopick_failed/ and the
same 105 recordings go through the manual nb25 sort (monday_janaug_2021_3slot) instead.
This script still defines the 105-recording slot set.

Writes a sort_log-shaped CSV (stem, decision, scene1_time_s) that
``build_frame_manifest.py`` consumes:

    .venv/bin/python3 scripts/build_autopick_janaug_2021.py
    .venv/bin/python3 scripts/build_frame_manifest.py \
        scene_sorting/spring_2021/autopick_log.csv \
        --out validation/monday_janaug_2021/frame_manifest.csv
    .venv/bin/python3 scripts/prelabel_manual_series.py \
        validation/monday_janaug_2021/frame_manifest.csv
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HARVEST = REPO / "notebooks/harvest_janaug2021_v8sahi.csv"
OUT = REPO / "scene_sorting/spring_2021/autopick_log.csv"

SLOTS = ("031500", "121500", "211500")
START, END = "2021-01-01", "2021-09-01"


def pick(rows: list[dict]) -> list[dict]:
    """Monday recordings in the fixed slots within [START, END), chronological."""
    out = []
    for r in rows:
        dt = datetime.fromisoformat(r["datetime_utc"])
        stem = r["frame_id"]
        if not (START <= dt.date().isoformat() < END):
            continue
        if dt.weekday() != 0 or stem[-6:] not in SLOTS:
            continue
        out.append(
            {"stem": stem, "decision": "scene1", "scene1_time_s": r["best_time_s"]}
        )
    return sorted(out, key=lambda r: r["stem"])


def main() -> None:
    with HARVEST.open(newline="") as fh:
        picks = pick(list(csv.DictReader(fh)))
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["stem", "decision", "scene1_time_s"])
        w.writeheader()
        w.writerows(picks)
    days = {p["stem"][10:18] for p in picks}
    print(f"wrote {OUT} -- {len(picks)} recordings over {len(days)} Mondays")


if __name__ == "__main__":
    main()

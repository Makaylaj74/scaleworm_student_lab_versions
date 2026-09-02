"""Extract full-res Scene-1 frames for validation rows once scene1_time_s is filled.

Run this AFTER a counter has opened the contact sheets for the -2021 rows and
written each recording's scene1_time_s into handcount_sheet.csv. It extracts the
full-resolution frame for every row that has a scene1_time_s but no ready frame
yet, and flips frame_ready to Y. Idempotent: already-extracted rows are skipped.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO = Path("/home/jovyan/scaleworm-student-lab")
sys.path.insert(0, str(REPO / "scripts"))
from count_frames import extract_frame

OUT = REPO / "validation/clear_window_handcount"
FRAMES = OUT / "frames"
SHEET_CSV = OUT / "handcount_sheet.csv"


def main() -> None:
    with SHEET_CSV.open() as fh:
        rows = list(csv.DictReader(fh))
        cols = rows[0].keys() if rows else []

    n = 0
    for r in rows:
        out = FRAMES / f"{r['frame_id']}.png"
        if r["frame_ready"] == "Y" and out.exists():
            continue
        if not r["scene1_time_s"]:
            continue  # still awaiting a Scene-1 pick from the contact sheet
        ok = extract_frame(Path(r["video_path"]), float(r["scene1_time_s"]), out)
        r["frame_ready"] = "Y" if (ok and out.exists()) else "extract-failed"
        n += 1

    with SHEET_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cols))
        w.writeheader()
        w.writerows(rows)
    print(f"extracted {n} newly-picked frames -> {FRAMES}")


if __name__ == "__main__":
    main()

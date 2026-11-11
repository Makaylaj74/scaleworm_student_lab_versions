"""Blind recount packet: one Sep-Dec 2021 Monday frame per week, fresh v2 pre-labels.

Why (2026-11-11): Jan-Aug 2021 box-corrected today reads 28/fr (MJ added +22/fr to v2's
5.8), while the Sep-Dec 2021 main series, corrected 2026-09-15, reads 10.6/fr (only +1.3/fr
added to v2's 9.3). Either the Sep+ close-up truly holds ~10 worms (or is a different patch),
or the Sep 2026 correction anchored on the pre-labels (annotator drift). Recounting a sample
of the Sep-Dec frames NOW with the identical protocol separates the two.

Blind: copies the image and a FRESH v2 pre-label (never MJ's old boxes), and blanks every
annotation column, so the old count is not visible in nb33.

    .venv/bin/python3 scripts/build_recount_sepdec_2021.py
"""

from __future__ import annotations

import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "validation/monday_manual_series"
OUT = REPO / "validation/recount_sepdec_2021"
V2 = REPO / "99_runs/scaleworm_v2/weights/best.pt"
CONF = 0.25
SEED = 20211111
BLANK = (
    "frame_status",
    "worm_count",
    "label_path",
    "annotation_method",
    "prelabel_model",
    "counter",
    "date_counted",
    "notes",
)


def main() -> None:
    from ultralytics import YOLO

    with (SRC / "frame_manifest.csv").open(newline="") as fh:
        reader = csv.DictReader(fh)
        fields, rows = reader.fieldnames, list(reader)
    by_day = defaultdict(list)
    for r in rows:
        if (
            r["frame_status"] == "counted"
            and "202109" <= r["frame_id"][10:16] <= "202112"
        ):
            by_day[r["frame_id"][10:18]].append(r)
    rng = random.Random(SEED)
    picks = [rng.choice(by_day[d]) for d in sorted(by_day)]

    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    model = YOLO(str(V2))
    out_rows = []
    for r in picks:
        fid = r["frame_id"]
        png = OUT / "images" / f"{fid}.png"
        shutil.copy2(SRC / "images" / f"{fid}.png", png)
        res = model(str(png), conf=CONF, verbose=False)[0]
        lines = [
            f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
            for x, y, w, h in res.boxes.xywhn.tolist()
        ]
        (OUT / "labels" / f"{fid}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )
        row = {**r, **{c: "" for c in BLANK}}
        row.update(
            frame_status="prelabeled",
            label_path=f"labels/{fid}.txt",
            prelabel_model="v2",
            notes="blind recount 2026-11 (drift check)",
        )
        out_rows.append(row)

    with (OUT / "frame_manifest.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {len(out_rows)} frames ({len(by_day)} Mondays) -> {OUT}")


if __name__ == "__main__":
    main()

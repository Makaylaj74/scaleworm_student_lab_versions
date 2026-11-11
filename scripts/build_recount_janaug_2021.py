"""Blind 8-frame recount of Jan-Aug 2021 under the confirmed counting rule (1 per month).

The Jan-Aug 2021 Monday frames were box-corrected on 2026-11-11 under an earlier broad
criterion (dark openings boxed even with no body visible), which read 2.35x the rule on
matched Sep-Dec 2021 Mondays. They are excluded from the series until recounted under
notebooks/worm_counting_rule.md. This quick pass recounts one counted frame per month
(seeded) with a FRESH v2 pre-label and blanked annotation columns. Same frames as the
broad pass, so the result is also a paired broad-vs-rule check for Jan-Aug.

    .venv/bin/python3 scripts/build_recount_janaug_2021.py
"""

from __future__ import annotations

import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "validation/monday_janaug_2021"
OUT = REPO / "validation/recount_janaug_2021"
V2 = REPO / "99_runs/scaleworm_v2/weights/best.pt"
CONF = 0.25
SEED = 20210101
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


def select(rows: list[dict], seed: int = SEED) -> list[dict]:
    """One counted frame per calendar month, chronological."""
    by_month: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["frame_status"] == "counted":
            by_month[r["frame_id"][10:16]].append(r)
    rng = random.Random(seed)
    return [rng.choice(by_month[m]) for m in sorted(by_month)]


def main() -> None:
    from ultralytics import YOLO

    done = OUT / "frame_manifest.csv"
    if done.exists() and "counted" in done.read_text():
        raise SystemExit(f"{done} already has counted frames -- refusing to overwrite")
    with (SRC / "frame_manifest.csv").open(newline="") as fh:
        reader = csv.DictReader(fh)
        fields, rows = reader.fieldnames, list(reader)
    picks = select(rows)

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
            notes="blind recount under worm_counting_rule.md (2026-11)",
        )
        out_rows.append(row)

    with done.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {len(out_rows)} frames -> {OUT}")


if __name__ == "__main__":
    main()

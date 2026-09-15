"""Build a per-camera-unit gate: pick N leakage-safe Scene-1 frames for one unit,
extract them, and lay out a handcount sheet for INDEPENDENT click-counting.

Given one or more Scene-1 ``sort_log.csv`` files and a camera unit (e.g.
``CAMHDA301-2017``), selects ~N evenly-spread scene1 frames belonging to that
unit (Aug(Y)->Aug(Y+1)), drops any whose day is in the v2/v3 training set
(leakage), extracts the full-res frame at ``scene1_time_s``, and writes:
  <out>/handcount_sheet.csv   (frame_id, datetime_utc, camera_unit, scene1_time_s,
                               worm_count[blank], counter, date_counted, notes)
  <out>/frames/<frame_id>.png
  <out>/clicks/               (empty; the click-counter fills it)

Workflow after this: open the click-counter (nb 29, SESSION repointed to <out>),
count each frame BLIND (clicks, no model boxes), then run score_unit_gate.py.

Env: N (default 25). Archive root fixed to the CAMHDA301 mp4 tree.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ARCHIVE = Path("/home/jovyan/ooi/san_data/RS03ASHS-PN03B-06-CAMHDA301")
N_DEFAULT = int(os.environ.get("N", "25"))
SWAP_MONTH = 8  # camera swapped each August -> unit Y spans Aug(Y)..Aug(Y+1)


def unit_of(dt: datetime) -> str:
    """Camera unit for a datetime: CAMHDA301-Y where Y=year if month>=Aug else year-1."""
    y = dt.year if dt.month >= SWAP_MONTH else dt.year - 1
    return f"CAMHDA301-{y}"


def even_spread(items: list, n: int) -> list:
    """Deterministic even-spread pick of n items across the list (keeps order)."""
    if n >= len(items):
        return items
    idx = [round(i * (len(items) - 1) / (n - 1)) for i in range(n)]
    return [items[i] for i in sorted(set(idx))]


def read_scene1(sort_logs: list[Path]) -> list[tuple[str, float]]:
    import pandas as pd

    out = []
    for log in sort_logs:
        d = pd.read_csv(log)
        s1 = d[d["decision"] == "scene1"]
        for _, r in s1.iterrows():
            out.append((r["stem"], float(r["scene1_time_s"])))
    return out


def training_days() -> set[str]:
    import glob

    days = set()
    for split in ("train", "val"):
        for p in glob.glob(
            str(
                Path(__file__).resolve().parent.parent
                / f"datasets/scaleworm_v2/labels/{split}/*.txt"
            )
        ):
            days.add(Path(p).stem.split("-")[1][:8])
    return days


def video_path(stem: str) -> Path:
    ts = stem.split("-")[1]
    dt = datetime.strptime(ts, "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
    return ARCHIVE / f"{dt.year}/{dt.month:02d}/{dt.day:02d}/{stem}.mp4"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sort_logs", nargs="+", type=Path)
    ap.add_argument("--unit", required=True, help="e.g. CAMHDA301-2017")
    ap.add_argument("--n", type=int, default=N_DEFAULT)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    tdays = training_days()
    rows = read_scene1(args.sort_logs)
    # filter to the requested unit, leakage-safe
    cand = []
    for stem, t in rows:
        dt = datetime.strptime(stem.split("-")[1], "%Y%m%dT%H%M%S").replace(
            tzinfo=UTC
        )
        if unit_of(dt) != args.unit:
            continue
        if stem.split("-")[1][:8] in tdays:
            continue
        cand.append((dt, stem, t))
    cand.sort()
    picked = even_spread(cand, args.n)
    print(
        f"{args.unit}: {len(cand)} leakage-safe scene1 frames -> picking {len(picked)}"
    )

    out = args.out
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "clicks").mkdir(parents=True, exist_ok=True)
    sheet = out / "handcount_sheet.csv"
    n_ok = 0
    with sheet.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "frame_id",
                "datetime_utc",
                "camera_unit",
                "scene1_time_s",
                "worm_count",
                "counter",
                "date_counted",
                "notes",
            ]
        )
        for dt, stem, t in picked:
            vid = video_path(stem)
            if not vid.exists():
                print(f"  missing video {stem}")
                continue
            png = out / "frames" / f"{stem}.png"
            if not png.exists():
                r = subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-ss",
                        str(t),
                        "-i",
                        str(vid),
                        "-frames:v",
                        "1",
                        "-q:v",
                        "2",
                        str(png),
                    ],
                    capture_output=True,
                    check=False,
                )
                if r.returncode != 0 or not png.exists():
                    print(f"  extract failed {stem}")
                    continue
            w.writerow([stem, dt.isoformat(), args.unit, t, "", "", "", ""])
            n_ok += 1
    print(
        f"wrote {sheet} ({n_ok} frames). Next: click-count BLIND (nb 29 -> {out}), "
        f"then: python scripts/score_unit_gate.py {out}"
    )


if __name__ == "__main__":
    main()

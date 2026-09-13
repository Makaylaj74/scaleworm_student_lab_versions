"""Manifest-driven pre-labeling for the unified Monday manual series.

Routes each pending Scene-1 frame in ``frame_manifest.csv`` to the appropriate
detector by recording date, extracts the Scene-1 frame, runs the model (conf=0.25),
and writes YOLO pre-labels for human correction. Updates the manifest in place
(``label_path``, ``prelabel_model``, ``frame_status``).

Routing (the Aug-2023 camera swap / blur onset is the boundary):
  * date <  BLUR_ONSET  -> v2  (clear-window specialist; gate recall ~86-103%)
  * date >= BLUR_ONSET  -> v3  (blurry post-2023 generalist; gate recall ~33%)

Pre-labels are a STARTING POINT, not ground truth. On clear frames v2 seeds most
boxes; on blurry frames v3 recovers only ~1/3, so the annotator ADDS many missed
worms there -- correction on the blurry stretch is closer to labelling from scratch.

Run on the GPU box (thesis venv, L40S). Resumable: frames that already have a label
file are skipped unless ``--force``. Never overwrites a frame already marked
``counted`` (human ground truth).

    <thesisvenv>/python scripts/prelabel_manual_series.py \
        validation/monday_manual_series/frame_manifest.csv

Env overrides: V2_MODEL, V3_MODEL (weight paths), CONF (default 0.25),
BLUR_ONSET (YYYY-MM-DD, default 2023-08-10).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import date, datetime
from pathlib import Path

REPO = Path("/home/jovyan/scaleworm-student-lab")
sys.path.insert(0, str(REPO / "scripts"))
from count_frames import extract_frame, parse_stem_dt  # noqa: E402 (needs sys.path insert)

CONF = float(os.environ.get("CONF", "0.25"))
BLUR_ONSET_DEFAULT = date(2023, 8, 10)  # -2023 camera swap ~Aug 10-11 2023 = blur onset
MODEL_PATHS = {
    "v2": Path(os.environ.get("V2_MODEL", str(REPO / "99_runs/scaleworm_v2/weights/best.pt"))),
    "v3": Path(os.environ.get("V3_MODEL", str(REPO / "99_runs/scaleworm_v3/weights/best.pt"))),
}
# frame_status values we must NOT overwrite (human-owned / terminal states).
PROTECTED_STATUS = frozenset({"counted", "no_scene1", "unusable_blur"})


def model_for_date(dt: datetime, blur_onset: date = BLUR_ONSET_DEFAULT) -> str:
    """Return which detector pre-labels a frame recorded at ``dt``."""
    return "v3" if dt.date() >= blur_onset else "v2"


def _load_models(needed: set[str]) -> dict:
    """Lazily import ultralytics and load only the models actually required."""
    from ultralytics import YOLO

    models = {}
    for key in needed:
        path = MODEL_PATHS[key]
        if not path.exists():
            raise FileNotFoundError(
                f"{key} weights not found at {path} "
                f"(set {key.upper()}_MODEL to override)"
            )
        models[key] = YOLO(str(path))
        print(f"loaded {key}: {path}")
    return models


def _prelabel_one(model, image_png: Path) -> list[str]:
    """Run the model on an extracted frame; return YOLO label lines."""
    res = model(str(image_png), conf=CONF, verbose=False)[0]
    return [
        f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for x, y, w, h in res.boxes.xywhn.tolist()
    ]


def _targets(rows: list[dict], labels_dir: Path, force: bool) -> list[dict]:
    """Rows to (re)pre-label: pending, not human-owned, not already done unless --force."""
    out = []
    for r in rows:
        if r.get("frame_status") in PROTECTED_STATUS:
            continue
        if not (r.get("scene1_time_s") or "").strip():
            print(f"  skip {r['frame_id']}: no scene1_time_s")
            continue
        if not force and (labels_dir / f"{r['frame_id']}.txt").exists():
            continue
        out.append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--force", action="store_true", help="re-pre-label frames that already have a label")
    ap.add_argument(
        "--blur-onset",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        default=BLUR_ONSET_DEFAULT,
        help="clear/blurry boundary date YYYY-MM-DD (default 2023-08-10)",
    )
    args = ap.parse_args()

    with args.manifest.open(newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        rows = list(reader)

    base = args.manifest.parent
    images_dir, labels_dir = base / "images", base / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    todo = _targets(rows, labels_dir, args.force)
    if not todo:
        print("nothing to pre-label (all done, protected, or unpicked). Use --force to redo.")
        return

    for r in todo:  # tag each row with its routed model before loading
        r["_route"] = model_for_date(parse_stem_dt(r["frame_id"]), args.blur_onset)
    needed = {r["_route"] for r in todo}
    print(f"{len(todo)} frames to pre-label: " + ", ".join(f"{k}={sum(r['_route'] == k for r in todo)}" for k in sorted(needed)))
    models = _load_models(needed)

    stats = {"v2": [0, 0], "v3": [0, 0]}  # model -> [frames, boxes]
    for r in todo:
        fid, route = r["frame_id"], r["_route"]
        png = images_dir / f"{fid}.png"
        if not png.exists() and not extract_frame(Path(r["video_path"]), float(r["scene1_time_s"]), png):
            print(f"  WARN {fid}: frame extraction failed (missing video?) — left pending")
            continue
        lines = _prelabel_one(models[route], png)
        (labels_dir / f"{fid}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        r["label_path"] = f"labels/{fid}.txt"
        r["prelabel_model"] = route
        r["frame_status"] = "prelabeled"
        stats[route][0] += 1
        stats[route][1] += len(lines)

    for r in rows:
        r.pop("_route", None)
    with args.manifest.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\nwrote pre-labels + updated {args.manifest}")
    for k in ("v2", "v3"):
        n, b = stats[k]
        if n:
            print(f"  {k}: {n} frames, {b} pre-boxes (mean {b / n:.1f}/frame)")
    print("NEXT: correct each frame in the box tool (ADD missed worms — heavy on the v3/blurry frames).")


if __name__ == "__main__":
    main()

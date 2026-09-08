"""Model-assisted pre-labels for the v2 retrain: extract frames + mushroom.pt boxes.

For each (stem, scene1_time_s, video_path) row in a manifest CSV, extract the Scene-1
frame and run mushroom.pt (conf=0.25); write the frame to <out>/images/<stem>.png and
its detections to <out>/labels/<stem>.txt in YOLO format (class 0 = scale_worm,
normalised cx cy w h).

These are PRE-labels, not ground truth. mushroom.pt has NO false positives but ~30-65%
recall, so a human corrects each frame by ADDING the missed worms (rarely deleting a
box). Correcting is far faster than labelling from scratch. After correction, move the
image+label into datasets/scaleworm_v2/images|labels/{train,val}.

    python scripts/make_prelabels.py datasets/scaleworm_v2/train_frames_2022.csv \
        datasets/scaleworm_v2/prelabels/2022
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from ultralytics import YOLO

REPO = Path("/home/jovyan/scaleworm-student-lab")
sys.path.insert(0, str(REPO / "scripts"))
from count_frames import extract_frame

CONF = 0.25


def main() -> None:
    manifest = Path(sys.argv[1])
    out = Path(sys.argv[2])
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    model = YOLO(str(REPO / "mushroom.pt"))
    rows = [
        r
        for r in csv.DictReader(manifest.open())
        if (r.get("scene1_time_s") or "").strip()  # skip un-picked rows
    ]
    n_box = 0
    for r in rows:
        stem = r.get("stem") or r["frame_id"]  # clean manifest or a pick_sheet
        png = out / "images" / f"{stem}.png"
        if not png.exists():
            extract_frame(Path(r["video_path"]), float(r["scene1_time_s"]), png)
        res = model(str(png), conf=CONF, verbose=False)[0]
        lines = [f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for x, y, w, h in res.boxes.xywhn.tolist()]
        (out / "labels" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        n_box += len(lines)
    print(f"{len(rows)} frames -> {out}  ({n_box} pre-boxes, mean {n_box / len(rows):.1f}/frame)")
    print("NEXT: correct in a labelling tool (add missed worms), then move to images|labels/{train,val}")


if __name__ == "__main__":
    main()

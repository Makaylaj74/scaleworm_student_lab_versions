"""Generate fixed-size YOLO pseudo-boxes from click points (weak labels).

The 2015-2016 early-recovery gate frames were click-counted (points), not box-labeled.
To use them as (weak) training data for an era fine-tune, drop a fixed-size box centered
on each click. Box size = the apparent worm size in this scene (~66 px: v5's own
detections median 68x65, Monday manual labels 60x58 at 1920x1080).

Output: YOLO label .txt per frame in <gate>/labels_pseudo/ (class 0), plus a manifest.
These are WEAK labels (fixed size, click-centered) -- fine for domain adaptation, not a
substitute for real box-correction.

Run: python scripts/generate_pseudo_boxes.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GATE = REPO / "validation/early_recovery_handcount_2015_2016"
CLICKS = GATE / "clicks"
OUT = GATE / "labels_pseudo"
BOX_W_PX = 66  # fixed apparent worm size (px) at 1920x1080
BOX_H_PX = 62


def main() -> None:
    sheet = [
        r for r in csv.DictReader((GATE / "handcount_sheet.csv").open())
        if (r.get("worm_count") or "").strip() not in ("", "nan")
    ]
    OUT.mkdir(exist_ok=True)
    total_boxes = 0
    written = 0
    manifest = []
    for r in sheet:
        fid = r["frame_id"]
        clk = CLICKS / f"{fid}.json"
        if not clk.exists():
            print(f"  skip {fid}: no clicks json")
            continue
        # frame size — all CamHD frames are 1920x1080; read defensively from the PNG
        import matplotlib.image as mpimg

        H, W = mpimg.imread(str(GATE / "frames" / f"{fid}.png")).shape[:2]
        wn, hn = BOX_W_PX / W, BOX_H_PX / H
        lines = []
        for px, py in json.load(clk.open())["clicks"]:
            cx, cy = px / W, py / H
            # clip center so the box stays in-frame
            cx = min(max(cx, wn / 2), 1 - wn / 2)
            cy = min(max(cy, hn / 2), 1 - hn / 2)
            lines.append(f"0 {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}")
        (OUT / f"{fid}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        total_boxes += len(lines)
        written += 1
        manifest.append({"frame_id": fid, "year": r["camera_unit"],
                         "n_boxes": len(lines), "manual_count": r["worm_count"]})
    with (GATE / "pseudo_box_manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame_id", "year", "n_boxes", "manual_count"])
        w.writeheader()
        w.writerows(manifest)
    print(f"wrote {written} pseudo-label files / {total_boxes} boxes -> {OUT}")
    print(f"box size {BOX_W_PX}x{BOX_H_PX}px  (normalized ~{BOX_W_PX / 1920:.4f} x {BOX_H_PX / 1080:.4f})")
    # sanity: pseudo boxes should equal manual clicks
    bad = [m for m in manifest if int(m["n_boxes"]) != int(m["manual_count"])]
    print("boxes==manual_count for all frames" if not bad else f"MISMATCH: {bad}")


if __name__ == "__main__":
    main()

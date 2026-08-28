"""Side-by-side: a sharp 2023 frame vs a blurry 2024 frame, each with mushroom.pt
detections (red boxes) overlaid and the human count stated. Shows directly why the
count collapses — on blurry footage the detector puts few/no boxes on worms that a
human still counts. Picks the sharpest 2023 hand-count frame and the blurriest 2024
hand-count frame that still has worms present (human_count >= 5).
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from ultralytics import YOLO

REPO = Path(__file__).resolve().parent.parent
OUT_PNG = REPO / "notebooks/figure_sharp_vs_blurry_detections.png"
CONF = 0.25
RED = "#D55E00"


def human_counts(csv_path):
    return {
        r["stem"]: int(r["human_count"])
        for r in csv.DictReader(csv_path.open())
        if r["human_count"].strip()
    }


def sharpness(png):
    g = cv2.cvtColor(cv2.imread(str(png)), cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(g, cv2.CV_64F).var()


def pick(frames_dir, truth, mode):
    """mode='sharp' -> max sharpness; 'blurry' -> min sharpness among worms-present frames."""
    cands = [
        (f, sharpness(f))
        for f in sorted(Path(frames_dir).glob("*.png"))
        if truth.get(f.stem, 0) >= 5
    ]
    return (max if mode == "sharp" else min)(cands, key=lambda t: t[1])


jan = human_counts(REPO / "notebooks/handcount_2023_01.csv")
may = human_counts(REPO / "notebooks/handcount_2024_05.csv")
sharp_f, sharp_v = pick(REPO / "notebooks/handcount_2023_01_frames", jan, "sharp")
blur_f, blur_v = pick(REPO / "notebooks/handcount_2024_05_frames", may, "blurry")

model = YOLO(str(REPO / "mushroom.pt"))
panels = [
    ("2023 — sharp footage", sharp_f, sharp_v, jan[sharp_f.stem], "#0072B2"),
    ("2024 — blurry footage", blur_f, blur_v, may[blur_f.stem], RED),
]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, (label, fpath, sval, hcount, color) in zip(axes, panels):
    img = cv2.cvtColor(cv2.imread(str(fpath)), cv2.COLOR_BGR2RGB)
    boxes = model.predict(str(fpath), conf=CONF, verbose=False)[0].boxes
    ax.imshow(img)
    for b in boxes:
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
        ax.add_patch(
            mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=RED, linewidth=2
            )
        )
    ax.set_axis_off()
    ax.set_title(
        f"{label}\nsharpness {sval:.0f}   ·   "
        f"model detected {len(boxes)}   ·   human counted {hcount}",
        fontsize=12,
        fontweight="bold",
        color=color,
    )

fig.suptitle(
    "Same worms, different footage: mushroom.pt detections (red) on a sharp vs a blurry frame",
    fontsize=13,
    fontweight="bold",
)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
print(f"sharp: {sharp_f.stem} sharpness={sharp_v:.0f} human={jan[sharp_f.stem]}")
print(f"blurry: {blur_f.stem} sharpness={blur_v:.0f} human={may[blur_f.stem]}")
print(f"wrote {OUT_PNG}")

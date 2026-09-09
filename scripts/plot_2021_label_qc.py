"""QC: overlay Makayla's corrected -2021 training boxes on their frames.

Draws the human-corrected YOLO boxes (green) on a spread of -2021 training frames
(sparsest first, plus a dense reference) so we can eyeball whether any clearly
visible worms were left un-boxed before spending GPU time on the retrain.

    /home/jovyan/joseph-scaleworm-thesis/.venv/bin/python scripts/plot_2021_label_qc.py OUT.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

REPO = Path("/home/jovyan/scaleworm-student-lab")
DS = REPO / "datasets/scaleworm_v2"
GREEN = "#009E73"

# sparsest -2021 frames (most likely to reveal misses) + one dense reference
STEMS = [
    "CAMHDA301-20220121T061500",  # 1 box
    "CAMHDA301-20211208T121500",  # 3 boxes
    "CAMHDA301-20220214T001500",  # 5 boxes
    "CAMHDA301-20211015T211500",  # 16 boxes (dense reference)
]


def _find(stem: str, kind: str, ext: str) -> Path | None:
    for split in ("train", "val"):
        p = DS / kind / split / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def main() -> None:
    out = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else REPO / "notebooks/figure_2021_label_qc.png"
    )
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for ax, stem in zip(axes.ravel(), STEMS):
        img_p = _find(stem, "images", ".png")
        lbl_p = _find(stem, "labels", ".txt")
        img = cv2.cvtColor(cv2.imread(str(img_p)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        ax.imshow(img)
        boxes = [
            line.split() for line in lbl_p.read_text().splitlines() if line.strip()
        ]
        for _cls, cx, cy, bw, bh in boxes:
            cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
            ax.add_patch(
                mpatches.Rectangle(
                    (cx - bw / 2, cy - bh / 2),
                    bw,
                    bh,
                    fill=False,
                    edgecolor=GREEN,
                    lw=2,
                )
            )
        ax.set_title(f"{stem}  —  {len(boxes)} boxes", fontsize=11)
        ax.axis("off")
    fig.suptitle(
        "QC: human-corrected -2021 training labels (green = worm boxes)", fontsize=13
    )
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

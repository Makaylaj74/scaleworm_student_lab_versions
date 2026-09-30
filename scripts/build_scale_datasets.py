"""Build nested dense-blurry training subsets for the Tier-B learning-curve test.

Question: does dense-blurry recall scale with more labels, or is it a ceiling? Reuse the
70 real Tuesday box-labels (no new manual work): train on nested subsets and gate each on
the SAME held-out 23 (datasets/scaleworm_v6_blurry/tuesday_blurry_test.txt).

Each subset dataset = the validated clear frames (reuse scaleworm_retrain_clear) + the first
N Tuesday-blurry TRAIN frames. Nested so N=15 ⊂ 30 ⊂ 47.

Run: python scripts/build_scale_datasets.py
"""

from __future__ import annotations

import csv
import glob
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CLEAR_DS = REPO / "datasets/scaleworm_retrain_clear"
TUE = REPO / "validation/tuesday_manual_series"
V6 = REPO / "datasets/scaleworm_v6_blurry"
SIZES = [15, 30, 47]


def main() -> None:
    test = set((V6 / "tuesday_blurry_test.txt").read_text().split())
    counted = [
        r["frame_id"]
        for r in csv.DictReader((TUE / "blurry_manifest.csv").open())
        if (r.get("worm_count") or "").strip() not in ("", "nan")
    ]
    train = sorted(s for s in counted if s not in test)  # deterministic, 47 frames
    print(f"Tuesday blurry: {len(counted)} counted, {len(test)} test, {len(train)} train pool")

    for n in SIZES:
        ds = REPO / f"datasets/scaleworm_v6_scale_n{n}"
        for sub in ("images/train", "images/val", "labels/train", "labels/val"):
            d = ds / sub
            if d.exists():
                for p in d.iterdir():
                    p.unlink()
            d.mkdir(parents=True, exist_ok=True)
        # clear (reuse retrain_clear splits verbatim)
        ntr = 0
        for split in ("train", "val"):
            for lbl in glob.glob(str(CLEAR_DS / "labels" / split / "*.txt")):
                stem = Path(lbl).stem
                img = REPO / "validation/monday_manual_series/images" / f"{stem}.png"
                (ds / "images" / split / f"{stem}.png").symlink_to(img.resolve())
                (ds / "labels" / split / f"{stem}.txt").symlink_to(Path(lbl).resolve())
                if split == "train":
                    ntr += 1
        # first N Tuesday-blurry into train
        for stem in train[:n]:
            (ds / "images/train" / f"{stem}.png").symlink_to((TUE / "images" / f"{stem}.png").resolve())
            (ds / "labels/train" / f"{stem}.txt").symlink_to((TUE / "labels" / f"{stem}.txt").resolve())
        (ds / "data.yaml").write_text(
            f"path: {ds}\ntrain: images/train\nval: images/val\nnames:\n  0: scale_worm\n"
        )
        print(f"  n{n}: {ntr} clear + {n} Tuesday-blurry train")


if __name__ == "__main__":
    main()

"""Build the v6 dataset: fix the blurry side with REAL Tuesday box-labels.

v5's blurry side was trained on under-counted Monday blurry labels (bad). v6 instead uses:
  - CLEAR   : the validated clear Monday frames (reuse datasets/scaleworm_retrain_clear)
  - BLURRY  : MJ's properly box-corrected Tuesday blurry frames
              (validation/tuesday_manual_series, worm_count == boxes, mean ~21/frame),
              day-level split so a held-out portion gates the blurry side.
The 161 under-counted Monday blurry frames are DROPPED.

Outputs datasets/scaleworm_v6_blurry/ (symlinks) + tuesday_blurry_test.txt (held-out gate).
Run: python scripts/build_v6_blurry_dataset.py
"""

from __future__ import annotations

import csv
import glob
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CLEAR_DS = REPO / "datasets/scaleworm_retrain_clear"  # already leakage-safe clear Monday
TUE = REPO / "validation/tuesday_manual_series"
DS = REPO / "datasets/scaleworm_v6_blurry"
SEED = 20260930
TEST_DAY_FRACTION = 0.30


def day_of(stem: str) -> str:
    return stem.split("-")[1][:8]


def main() -> None:
    # Tuesday blurry frames that MJ actually box-counted
    tue = [
        r["frame_id"]
        for r in csv.DictReader((TUE / "blurry_manifest.csv").open())
        if (r.get("worm_count") or "").strip() not in ("", "nan")
        and (TUE / "labels" / f"{r['frame_id']}.txt").exists()
        and (TUE / "images" / f"{r['frame_id']}.png").exists()
    ]
    # day-level split of the Tuesday blurry frames
    days = sorted({day_of(s) for s in tue})
    rng = random.Random(SEED)
    rng.shuffle(days)
    test_days = set(days[: max(1, round(len(days) * TEST_DAY_FRACTION))])
    tue_train = [s for s in tue if day_of(s) not in test_days]
    tue_test = [s for s in tue if day_of(s) in test_days]

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        d = DS / sub
        if d.exists():
            for p in d.iterdir():
                p.unlink()
        d.mkdir(parents=True, exist_ok=True)

    n = {"train": 0, "val": 0}
    # 1) clear: reuse the leakage-safe clear dataset's splits verbatim
    for split in ("train", "val"):
        for lbl in glob.glob(str(CLEAR_DS / "labels" / split / "*.txt")):
            stem = Path(lbl).stem
            img = CLEAR_DS / "images" / split / f"{stem}.png"
            if not img.exists():
                # resolve through the symlink source
                img = REPO / "validation/monday_manual_series/images" / f"{stem}.png"
            (DS / "images" / split / f"{stem}.png").symlink_to(img.resolve())
            (DS / "labels" / split / f"{stem}.txt").symlink_to(Path(lbl).resolve())
            n[split] += 1
    # 2) blurry train: Tuesday train frames (a few to val for early stop)
    rng.shuffle(tue_train)
    n_val = max(2, round(len(tue_train) * 0.15))
    for i, stem in enumerate(tue_train):
        split = "val" if i < n_val else "train"
        (DS / "images" / split / f"{stem}.png").symlink_to((TUE / "images" / f"{stem}.png").resolve())
        (DS / "labels" / split / f"{stem}.txt").symlink_to((TUE / "labels" / f"{stem}.txt").resolve())
        n[split] += 1

    (DS / "data.yaml").write_text(
        f"path: {DS}\ntrain: images/train\nval: images/val\nnames:\n  0: scale_worm\n"
    )
    (DS / "tuesday_blurry_test.txt").write_text("\n".join(sorted(tue_test)) + "\n")

    # leakage assertion: no Tuesday test frame in train/val
    train_val = {Path(p).stem for p in glob.glob(str(DS / "labels/train/*.txt"))} | {
        Path(p).stem for p in glob.glob(str(DS / "labels/val/*.txt"))
    }
    assert not (set(tue_test) & train_val), "Tuesday test leaked into train/val"

    print(f"Tuesday blurry: {len(tue)} counted -> {len(tue_train)} train / {len(tue_test)} test "
          f"({len(test_days)}/{len(days)} days held out)")
    print(f"v6_blurry dataset: {n['train']} train / {n['val']} val  (clear reused + Tuesday blurry)")
    print(f"held-out gate: {len(tue_test)} Tuesday-blurry frames + the 39 clear-click frames")
    print("leakage check PASSED")
    print(f"wrote {DS}")


if __name__ == "__main__":
    main()

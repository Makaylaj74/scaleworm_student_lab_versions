"""Build the tiny 2015-2016 domain fine-tune dataset from pseudo-boxes.

27 click-counted early-recovery frames with fixed-box weak labels
(labels_pseudo/). Year-stratified deterministic split so 2015 and 2016 both appear in
every split; the TEST frames are held out for a click-based gate (scored separately via
their clicks, NOT the pseudo-boxes). Symlinks images + pseudo-labels into
datasets/scaleworm_v6_early2015/.

Split ~ 55% train / 15% val (early stop) / 30% test.
Run: python scripts/build_early_finetune_dataset.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GATE = REPO / "validation/early_recovery_handcount_2015_2016"
LABELS = GATE / "labels_pseudo"
DS = REPO / "datasets/scaleworm_v6_early2015"


def main() -> None:
    rows = [
        r for r in csv.DictReader((GATE / "handcount_sheet.csv").open())
        if (r.get("worm_count") or "").strip() not in ("", "nan")
        and (LABELS / f"{r['frame_id']}.txt").exists()
    ]
    by_year: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        by_year[r["camera_unit"]].append(r["frame_id"])

    train, val, test = [], [], []
    for stems in by_year.values():
        stems = sorted(stems)  # deterministic
        n = len(stems)
        n_test = max(1, round(n * 0.30))
        n_val = max(1, round(n * 0.15))
        # stride the test/val picks across the year so splits span months, not contiguous
        test += stems[::3][:n_test]
        rest = [s for s in stems if s not in set(stems[::3][:n_test])]
        val += rest[:n_val]
        train += rest[n_val:]

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        d = DS / sub
        if d.exists():
            for p in d.iterdir():
                p.unlink()
        d.mkdir(parents=True, exist_ok=True)

    for split, stems in (("train", train), ("val", val)):
        for s in stems:
            (DS / "images" / split / f"{s}.png").symlink_to(GATE / "frames" / f"{s}.png")
            (DS / "labels" / split / f"{s}.txt").symlink_to(LABELS / f"{s}.txt")

    (DS / "data.yaml").write_text(
        f"path: {DS}\ntrain: images/train\nval: images/val\nnames:\n  0: scale_worm\n"
    )
    (DS / "test_frames.txt").write_text("\n".join(test) + "\n")
    print(f"train {len(train)} / val {len(val)} / test {len(test)} (held-out gate)")
    print(f"  train: {sorted(train)}")
    print(f"  test : {sorted(test)}")
    print(f"wrote {DS}")


if __name__ == "__main__":
    main()

"""Carve a deterministic val split from the v2 training frames.

Moves an even-spread ~FRACTION of the corrected train image/label pairs into
images/val + labels/val. Deterministic (sorted stems, fixed stride) and idempotent:
re-running restores the intended split regardless of current train/val placement.

The 39-frame clear-window gate is the INDEPENDENT acceptance test and is never touched
here — this split only holds out a slice of the training frames for YOLO's own val metric.

    python scripts/make_val_split.py            # default 15% val
    VAL_FRACTION=0.2 python scripts/make_val_split.py
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path("/home/jovyan/scaleworm-student-lab/datasets/scaleworm_v2")
FRACTION = float(os.environ.get("VAL_FRACTION", "0.15"))


def val_stems(all_stems: list[str], fraction: float) -> set[str]:
    """Pick the val subset: even-spread across the SORTED stem list, deterministic (no RNG).

    Returns at least one stem for any non-empty input. Idempotent — the same inputs always
    select the same stems, so re-running the split is stable.
    """
    stems = sorted(all_stems)
    if not stems:
        return set()
    n_val = max(1, round(len(stems) * fraction))
    step = len(stems) / n_val
    return {stems[int(i * step)] for i in range(n_val)}


def _pairs(split: str) -> set[str]:
    return {p.stem for p in (ROOT / "labels" / split).glob("*.txt")}


def _move(stem: str, src: str, dst: str) -> None:
    for kind, ext in (("images", ".png"), ("labels", ".txt")):
        s = ROOT / kind / src / f"{stem}{ext}"
        d = ROOT / kind / dst / f"{stem}{ext}"
        d.parent.mkdir(parents=True, exist_ok=True)
        if s.exists():
            shutil.move(str(s), str(d))


def main() -> None:
    all_stems = sorted(_pairs("train") | _pairs("val"))
    val = val_stems(all_stems, FRACTION)

    for stem in all_stems:
        want = "val" if stem in val else "train"
        cur = "val" if stem in _pairs("val") else "train"
        if cur != want:
            _move(stem, cur, want)

    print(f"total={len(all_stems)}  val={len(val)}  train={len(all_stems) - len(val)}")
    print("val frames:", ", ".join(sorted(val)))


if __name__ == "__main__":
    main()

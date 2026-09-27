"""Tests for scripts/irr_counts_agreement.py (count/box IRR)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import irr_counts_agreement as irr


def _write_boxes(path: Path, centers):
    """Write a YOLO label file with fixed-size boxes at the given (cx, cy) centers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"0 {cx:.4f} {cy:.4f} 0.05 0.05" for cx, cy in centers]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def _make_rater(root: Path, name: str, frames: dict):
    """frames: {frame_id: (status, [(cx,cy), ...])}. Builds manifest + labels/."""
    rdir = root / name
    (rdir / "labels").mkdir(parents=True)
    fields = ["frame_id", "frame_status", "worm_count", "counter"]
    rows = []
    for fid, (status, centers) in frames.items():
        if centers is not None:
            _write_boxes(rdir / "labels" / f"{fid}.txt", centers)
        rows.append({"frame_id": fid, "frame_status": status,
                     "worm_count": str(len(centers)) if centers is not None else "",
                     "counter": name})
    with (rdir / "irr_manifest.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return rdir


def test_load_boxes(tmp_path):
    p = tmp_path / "f.txt"
    _write_boxes(p, [(0.1, 0.2), (0.3, 0.4)])
    boxes = irr.load_boxes(p)
    assert len(boxes) == 2
    assert boxes[0] == pytest.approx((0.1, 0.2, 0.05, 0.05))


def test_load_boxes_missing(tmp_path):
    assert irr.load_boxes(tmp_path / "nope.txt") == []


def test_match_boxes_all_hit():
    a = [(0.1, 0.1, 0.05, 0.05), (0.5, 0.5, 0.05, 0.05)]
    b = [(0.1, 0.1, 0.05, 0.05), (0.5, 0.5, 0.05, 0.05)]
    assert irr.match_boxes(a, b) == 2


def test_match_boxes_partial():
    a = [(0.1, 0.1, 0.05, 0.05), (0.9, 0.9, 0.05, 0.05)]  # 2nd is far away
    b = [(0.1, 0.1, 0.05, 0.05)]
    assert irr.match_boxes(a, b) == 1


def test_match_boxes_one_claim_each():
    # two A centres inside one B box -> only one match
    a = [(0.50, 0.50, 0.02, 0.02), (0.51, 0.51, 0.02, 0.02)]
    b = [(0.50, 0.50, 0.10, 0.10)]
    assert irr.match_boxes(a, b) == 1


def test_icc_perfect_agreement():
    m = np.array([[5, 5], [10, 10], [2, 2]], dtype=float)
    assert irr.icc_a1(m) == pytest.approx(1.0)


def test_icc_all_identical_rows():
    # zero total variance -> defined as perfect
    m = np.array([[3, 3], [3, 3]], dtype=float)
    assert irr.icc_a1(m) == 1.0


def test_icc_disagreement_lowers():
    good = np.array([[5, 5], [10, 10], [2, 3]], dtype=float)
    bad = np.array([[5, 1], [10, 2], [2, 9]], dtype=float)
    assert irr.icc_a1(good) > irr.icc_a1(bad)


def test_icc_degenerate():
    assert np.isnan(irr.icc_a1(np.array([[1.0]])))


def test_compute_two_raters(tmp_path):
    frames_a = {
        "CAMHDA301-20220117T001500": ("counted", [(0.1, 0.1), (0.2, 0.2)]),
        "CAMHDA301-20220117T031500": ("counted", [(0.5, 0.5)]),
        "CAMHDA301-20220117T121500": ("unusable_blur", None),  # excluded
    }
    frames_b = {
        "CAMHDA301-20220117T001500": ("counted", [(0.1, 0.1), (0.9, 0.9)]),  # 1 match
        "CAMHDA301-20220117T031500": ("counted", [(0.5, 0.5)]),  # exact
        "CAMHDA301-20220117T121500": ("counted", [(0.3, 0.3)]),  # a excluded it
    }
    a = _make_rater(tmp_path, "MJ", frames_a)
    b = _make_rater(tmp_path, "LG", frames_b)
    res = irr.compute([a, b], n_boot=200)

    assert res["raters"] == ["MJ", "LG"]
    assert res["n_joint"] == 2  # the blur frame drops out
    assert res["count"]["totals"] == {"MJ": 3, "LG": 3}
    # frame 2 exact (1==1); frame 1 is 2 vs 2 -> also exact count
    assert res["count"]["exact_all"] == pytest.approx(1.0)
    pair = res["count"]["pairwise"]["MJ|LG"]
    assert pair["mean_abs_diff"] == pytest.approx(0.0)
    box = res["box"]["pairwise"]["MJ|LG"]
    # frame1: 1 of 2 boxes match; frame2: 1 of 1 -> micro F1 = 2*2/(3+3)
    assert box["micro_f1"] == pytest.approx(2 * 2 / (3 + 3))


def test_compute_requires_two(tmp_path):
    a = _make_rater(tmp_path, "MJ", {"f": ("counted", [(0.1, 0.1)])})
    with pytest.raises(ValueError):
        irr.compute([a])


def test_compute_no_joint(tmp_path):
    a = _make_rater(tmp_path, "MJ", {"f1": ("counted", [(0.1, 0.1)])})
    b = _make_rater(tmp_path, "LG", {"f2": ("counted", [(0.1, 0.1)])})
    res = irr.compute([a, b], n_boot=50)
    assert res["n_joint"] == 0
    assert res["count"] == {}

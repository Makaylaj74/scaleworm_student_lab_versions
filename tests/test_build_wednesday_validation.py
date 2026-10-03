"""Tests for scripts/build_wednesday_validation.py (stratified Wednesday sampler)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_wednesday_validation as bwv


def _rows():
    rows = []
    for year in ("2021", "2022"):
        for slot in ("001500", "031500"):
            for i in range(5):  # 5 recordings per (year, slot) stratum
                rows.append({"stem": f"CAMHDA301-{year}010{i + 1}T{slot}", "slot": slot,
                             "video_exists": "True"})
    return rows


def test_video_path_parses_stem():
    p = bwv.video_path("CAMHDA301-20210908T001500")
    assert p.as_posix().endswith("2021/09/08/CAMHDA301-20210908T001500.mp4")


def test_stratified_draw_per_stratum_count():
    picked = bwv.stratified_draw(_rows(), per_stratum=3, seed=1)
    # 2 years x 2 slots = 4 strata, 3 each
    assert len(picked) == 12
    from collections import Counter
    strata = Counter((r["stem"].split("-")[1][:4], r["slot"]) for r in picked)
    assert set(strata.values()) == {3}


def test_stratified_draw_caps_at_available():
    rows = _rows()[:2]  # only 2 in one stratum
    picked = bwv.stratified_draw(rows, per_stratum=3, seed=1)
    assert len(picked) == 2  # cannot exceed what's available


def test_stratified_draw_deterministic():
    a = bwv.stratified_draw(_rows(), per_stratum=2, seed=7)
    b = bwv.stratified_draw(_rows(), per_stratum=2, seed=7)
    assert [r["stem"] for r in a] == [r["stem"] for r in b]


def test_stratified_draw_sorted_output():
    picked = bwv.stratified_draw(_rows(), per_stratum=3, seed=3)
    stems = [r["stem"] for r in picked]
    assert stems == sorted(stems)

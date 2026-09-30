"""Tests for the scene-sort-free harvest pipeline."""

import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import build_harvest_manifest as bhm
import harvest_series as hs


def test_parse_weekdays():
    assert bhm.parse_weekdays("1-4") == {1, 2, 3, 4}
    assert bhm.parse_weekdays("0,2,4") == {0, 2, 4}
    assert bhm.parse_weekdays("0-4") == {0, 1, 2, 3, 4}


def test_dt_from_stem_utc():
    dt = bhm.dt_from_stem("CAMHDA301-20230307T061500")
    assert dt == datetime(2023, 3, 7, 6, 15, 0, tzinfo=UTC)


def test_canonical_slots_membership():
    # dual-family: 2017+ record at HH:15:00, 2015-2016 at HH:00:00
    assert "061500" in bhm.CANONICAL_SLOTS  # 2017+ slot
    assert "060000" in bhm.CANONICAL_SLOTS  # 2015-2016 slot
    assert "223000" not in bhm.CANONICAL_SLOTS  # non-standard time filtered
    assert len(bhm.CANONICAL_SLOTS) == 16  # 8 slots x 2 minute-families


def test_aggregate_median_of_top3():
    # counts across the pan window: front-on frames are the richest
    tc = [(120.0, 5), (180.0, 20), (240.0, 18), (300.0, 3)]
    out = hs.aggregate(tc)
    assert out["max_count"] == 20
    assert out["count_raw"] == 18  # median of top-3 [20,18,5]
    assert out["best_time_s"] == 240.0  # time of the median member
    assert out["n_frames"] == 4
    assert out["count_corrected"] == round(18 / hs.RECALL, 2)


def test_aggregate_fewer_than_three():
    out = hs.aggregate([(120.0, 7), (180.0, 4)])
    assert out["count_raw"] == 5  # int(median([7,4])=5.5) = 5
    assert out["max_count"] == 7
    assert out["n_frames"] == 2

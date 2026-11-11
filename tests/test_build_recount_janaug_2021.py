"""Tests for the Jan-Aug 2021 rule-recount frame selection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_recount_janaug_2021 import rest, select


def _row(day: str, slot: str = "211500", status: str = "counted") -> dict:
    return {"frame_id": f"CAMHDA301-{day}T{slot}", "frame_status": status}


def _pool() -> list[dict]:
    rows = [
        _row(f"2021{m:02d}{d:02d}", s)
        for m in (1, 2, 3)
        for d in (4, 11)
        for s in ("031500", "211500")
    ]
    return rows + [_row("20210118", status="unusable_blur")]


def test_select_one_counted_frame_per_month():
    picks = select(_pool(), seed=3)
    assert [r["frame_id"][10:16] for r in picks] == ["202101", "202102", "202103"]
    assert all(r["frame_status"] == "counted" for r in picks)


def test_rest_is_complement_of_sample_and_skips_blur():
    pool = _pool()
    sampled = {r["frame_id"] for r in select(pool, seed=3)}
    got = rest(pool, sampled)
    counted = {r["frame_id"] for r in pool if r["frame_status"] == "counted"}
    assert {r["frame_id"] for r in got} == counted - sampled
    assert [r["frame_id"] for r in got] == sorted(r["frame_id"] for r in got)

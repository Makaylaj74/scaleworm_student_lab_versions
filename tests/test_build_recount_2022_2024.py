"""Tests for the stratified blind recount selection."""

import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_recount_2022_2024 import STRATA, select, stratum_of


def _row(day: str, slot: str = "121500", status: str = "counted") -> dict:
    return {"frame_id": f"CAMHDA301-{day}T{slot}", "frame_status": status}


def test_stratum_boundaries():
    assert stratum_of("20220630") == "2022_h1"
    assert stratum_of("20220701") == "2022_h2_peak"
    assert stratum_of("20230809") == "2023_clear"
    assert stratum_of("20230810") == "2023_blurry"
    assert stratum_of("20241231") == "2024"
    assert stratum_of("20211231") is None


def _pool() -> list[dict]:
    """Ten Mondays-worth of days per stratum, two slots each, plus noise rows."""
    rows = []
    for lo, _, _ in STRATA.values():
        start = date(int(lo[:4]), int(lo[4:6]), int(lo[6:]))
        for d in range(10):
            day = (start + timedelta(days=d)).strftime("%Y%m%d")
            rows += [_row(day), _row(day, "211500")]
    rows.append(_row("20220103", status="pending"))
    return rows


def test_select_one_frame_per_monday_and_n_per_stratum():
    picks = select(_pool(), exclude=set(), seed=1)
    for name, (_, _, n) in STRATA.items():
        got = [r for s, r in picks if s == name]
        assert len(got) == n
        assert len({r["frame_id"][10:18] for r in got}) == n  # distinct days
    assert all(r["frame_status"] == "counted" for _, r in picks)


def test_select_respects_exclusions_and_is_deterministic():
    pool = _pool()
    first = select(pool, exclude=set(), seed=7)
    excluded = {r["frame_id"] for _, r in first}
    second = select(pool, exclude=excluded, seed=7)
    assert not excluded & {r["frame_id"] for _, r in second}
    assert select(pool, exclude=set(), seed=7) == first

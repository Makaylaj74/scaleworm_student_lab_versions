"""Tests for the per-unit gate selection logic (build_unit_gate)."""

import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_unit_gate import even_spread, unit_of


def test_unit_of_august_boundary():
    # camera swaps in August: unit Y spans Aug(Y) .. Aug(Y+1)
    utc = UTC
    assert unit_of(datetime(2017, 8, 15, tzinfo=utc)) == "CAMHDA301-2017"
    assert unit_of(datetime(2017, 9, 1, tzinfo=utc)) == "CAMHDA301-2017"
    assert unit_of(datetime(2018, 7, 31, tzinfo=utc)) == "CAMHDA301-2017"
    assert unit_of(datetime(2017, 7, 31, tzinfo=utc)) == "CAMHDA301-2016"
    assert unit_of(datetime(2017, 1, 1, tzinfo=utc)) == "CAMHDA301-2016"


def test_even_spread_returns_all_when_n_large():
    items = [1, 2, 3]
    assert even_spread(items, 5) == [1, 2, 3]
    assert even_spread(items, 3) == [1, 2, 3]


def test_even_spread_hits_endpoints_and_count():
    items = list(range(100))
    out = even_spread(items, 10)
    assert out[0] == 0
    assert out[-1] == 99
    assert len(out) == 10
    # strictly increasing (no duplicates)
    assert out == sorted(set(out))


def test_even_spread_preserves_order():
    items = list("abcdefghij")
    out = even_spread(items, 4)
    assert out == sorted(out, key=items.index)

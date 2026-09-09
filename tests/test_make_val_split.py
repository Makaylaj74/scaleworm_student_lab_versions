"""Tests for the deterministic val-split selection in scripts/make_val_split.py."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from make_val_split import val_stems  # noqa: E402  (import needs sys.path insert above)


def test_fraction_and_count():
    stems = [f"f{i:02d}" for i in range(40)]
    val = val_stems(stems, 0.15)
    assert len(val) == 6  # round(40 * 0.15)
    assert val <= set(stems)


def test_deterministic():
    stems = [f"f{i:02d}" for i in range(40)]
    assert val_stems(stems, 0.15) == val_stems(stems, 0.15)


def test_order_independent():
    stems = [f"f{i:02d}" for i in range(40)]
    assert val_stems(stems, 0.15) == val_stems(list(reversed(stems)), 0.15)


def test_even_spread_not_clustered():
    # even-spread should not put all val stems at the very start of the sorted list
    stems = [f"f{i:02d}" for i in range(40)]
    val = val_stems(stems, 0.15)
    idx = sorted(stems.index(s) for s in val)
    assert idx[-1] - idx[0] > len(stems) // 2  # spans well beyond the first half


def test_nonempty_input_yields_at_least_one():
    assert len(val_stems(["only"], 0.01)) == 1


def test_empty_input():
    assert val_stems([], 0.15) == set()

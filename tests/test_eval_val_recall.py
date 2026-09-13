"""Tests for the center-based matcher in eval_val_recall."""

from scripts.eval_val_recall import center_match


def _p(cx, cy, conf=0.5, s=2.0):
    """A prediction box centered at (cx, cy)."""
    return (cx - s, cy - s, cx + s, cy + s, conf)


def _g(cx, cy, s=5.0):
    """A GT box centered at (cx, cy)."""
    return (cx - s, cy - s, cx + s, cy + s)


def test_perfect_match():
    gts = [_g(10, 10), _g(50, 50)]
    preds = [_p(10, 10), _p(50, 50)]
    assert center_match(preds, gts) == (2, 0, 0)


def test_missed_worm_is_fn():
    gts = [_g(10, 10), _g(50, 50)]
    preds = [_p(10, 10)]
    assert center_match(preds, gts) == (1, 0, 1)  # one TP, one missed


def test_false_positive():
    gts = [_g(10, 10)]
    preds = [_p(10, 10), _p(200, 200)]
    assert center_match(preds, gts) == (1, 1, 0)


def test_each_gt_claimed_once():
    # two predictions on the same worm -> 1 TP, 1 FP (not 2 TP)
    gts = [_g(10, 10)]
    preds = [_p(10, 10, conf=0.9), _p(11, 11, conf=0.8)]
    assert center_match(preds, gts) == (1, 1, 0)


def test_empty_predictions_all_fn():
    gts = [_g(10, 10), _g(50, 50)]
    assert center_match([], gts) == (0, 0, 2)


def test_empty_gt_all_fp():
    preds = [_p(10, 10), _p(50, 50)]
    assert center_match(preds, []) == (0, 2, 0)


def test_center_outside_box_is_not_a_hit():
    gts = [_g(10, 10, s=1.0)]  # tiny box 9..11
    preds = [_p(20, 20)]  # center far outside
    assert center_match(preds, gts) == (0, 1, 1)

"""Tests for the point-in-box matcher in eval_clear_recall."""

from scripts.eval_clear_recall import point_match


def _p(x1, y1, x2, y2, conf=0.5):
    return (x1, y1, x2, y2, conf)


def test_all_clicks_hit():
    preds = [_p(0, 0, 10, 10), _p(20, 20, 30, 30)]
    points = [(5, 5), (25, 25)]
    assert point_match(preds, points) == (2, 0, 0)


def test_missed_click_is_fn():
    preds = [_p(0, 0, 10, 10)]
    points = [(5, 5), (100, 100)]
    assert point_match(preds, points) == (1, 0, 1)


def test_box_with_no_click_is_fp():
    preds = [_p(0, 0, 10, 10), _p(50, 50, 60, 60)]
    points = [(5, 5)]
    assert point_match(preds, points) == (1, 1, 0)


def test_two_boxes_one_click_one_fp():
    # two overlapping boxes on the same worm -> only one claims the click
    preds = [_p(0, 0, 10, 10, conf=0.9), _p(1, 1, 9, 9, conf=0.8)]
    points = [(5, 5)]
    assert point_match(preds, points) == (1, 1, 0)


def test_one_box_two_clicks_claims_one():
    # a single box covering two clicks only claims one (one box = one detection)
    preds = [_p(0, 0, 100, 100)]
    points = [(5, 5), (50, 50)]
    assert point_match(preds, points) == (1, 0, 1)


def test_no_predictions():
    assert point_match([], [(5, 5), (9, 9)]) == (0, 0, 2)


def test_no_clicks():
    assert point_match([_p(0, 0, 10, 10)], []) == (0, 1, 0)

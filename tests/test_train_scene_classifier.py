"""Unit tests for the scene-classifier metric helpers (no torch training run)."""

import numpy as np
import pytest

from scripts.train_scene_classifier import best_f1_threshold, prf, roc_auc


def test_prf_perfect_separation():
    labels = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.2, 0.8, 0.9])
    m = prf(labels, probs, thr=0.5)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 0, 0, 2)


def test_prf_threshold_shifts_counts():
    labels = np.array([0, 1, 1])
    probs = np.array([0.4, 0.4, 0.9])
    # thr=0.5 -> only the 0.9 fires: tp=1 (the 0.9), fn=1 (the 0.4 pos), the 0.4 neg is tn
    m = prf(labels, probs, thr=0.5)
    assert (m["tp"], m["fn"], m["tn"], m["fp"]) == (1, 1, 1, 0)


def test_roc_auc_perfect_and_random():
    labels = np.array([0, 0, 1, 1])
    assert roc_auc(labels, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    # fully reversed -> AUC 0
    assert roc_auc(labels, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0


def test_roc_auc_ties_give_half():
    labels = np.array([0, 1])
    # identical scores -> chance
    assert roc_auc(labels, np.array([0.5, 0.5])) == pytest.approx(0.5)


def test_roc_auc_degenerate_single_class():
    assert np.isnan(roc_auc(np.array([1, 1]), np.array([0.3, 0.7])))


def test_best_f1_threshold_recovers_separator():
    labels = np.array([0, 0, 1, 1])
    probs = np.array([0.2, 0.3, 0.7, 0.8])
    thr, f1 = best_f1_threshold(labels, probs)
    assert f1 == 1.0
    assert 0.3 < thr <= 0.7

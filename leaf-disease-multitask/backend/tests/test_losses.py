"""Loss / metric math tests. Need TensorFlow; skipped if not installed."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tensorflow")

import tensorflow as tf  # noqa: E402

from app import losses  # noqa: E402


def _mask(batch):
    return tf.constant(np.array(batch, dtype=np.float32)[..., None])


def test_perfect_prediction_scores_one():
    y = _mask([[[1, 0], [0, 1]]])
    iou = losses.iou_score(y, y).numpy()
    dice = losses.dice_coefficient(y, y).numpy()
    assert iou.shape == (1,)
    assert pytest.approx(iou[0], abs=1e-4) == 1.0
    assert pytest.approx(dice[0], abs=1e-4) == 1.0


def test_disjoint_prediction_scores_low():
    # Larger masks so the Dice/IoU smoothing term is negligible.
    top = np.ones((10, 10), np.float32)
    bottom = np.ones((10, 10), np.float32)
    y_true = _mask([np.vstack([top, np.zeros_like(top)])])      # leaf on top half
    y_pred = _mask([np.vstack([np.zeros_like(bottom), bottom])])  # leaf on bottom half
    iou = losses.iou_score(y_true, y_pred).numpy()[0]
    # No overlap -> IoU close to 0.
    assert iou < 0.05


def test_dice_bce_loss_is_per_sample():
    y_true = _mask([[[1, 0], [0, 1]], [[0, 0], [0, 0]]])
    y_pred = _mask([[[1, 0], [0, 1]], [[1, 1], [1, 1]]])
    loss = losses.dice_bce_loss(y_true, y_pred).numpy()
    assert loss.shape == (2,)
    # First sample is a perfect match -> lower loss than the wrong second one.
    assert loss[0] < loss[1]


def test_loss_non_negative():
    rng = np.random.default_rng(0)
    y_true = _mask(rng.integers(0, 2, size=(3, 8, 8)))
    y_pred = tf.constant(rng.random((3, 8, 8, 1)).astype(np.float32))
    loss = losses.dice_bce_loss(y_true, y_pred).numpy()
    assert np.all(loss >= 0.0)

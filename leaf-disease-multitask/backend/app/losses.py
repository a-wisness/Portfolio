"""Segmentation loss and mask metrics (IoU, Dice).

All functions return a **per-sample** tensor of shape ``(batch,)`` so they work
correctly as Keras ``weighted_metrics`` / losses: Keras then applies the
per-sample segmentation ``sample_weight`` (0 for classification-only images, 1
for segmentation images) and reduces. This is what lets one model train on the
two disjoint datasets at once.
"""
from __future__ import annotations

import tensorflow as tf

_SMOOTH = 1.0


def _per_sample_sums(y_true: tf.Tensor, y_pred: tf.Tensor):
    """Flatten H,W,C and return per-sample (intersection, sum_true, sum_pred)."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    axes = [1, 2, 3]  # reduce spatial + channel, keep batch
    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    sum_true = tf.reduce_sum(y_true, axis=axes)
    sum_pred = tf.reduce_sum(y_pred, axis=axes)
    return intersection, sum_true, sum_pred


def dice_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Soft Dice per sample, shape ``(batch,)``."""
    intersection, sum_true, sum_pred = _per_sample_sums(y_true, y_pred)
    return (2.0 * intersection + _SMOOTH) / (sum_true + sum_pred + _SMOOTH)


def iou_score(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Intersection-over-Union per sample on a 0.5-thresholded mask, ``(batch,)``."""
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
    intersection, sum_true, sum_pred = _per_sample_sums(y_true, y_pred_bin)
    union = sum_true + sum_pred - intersection
    return (intersection + _SMOOTH) / (union + _SMOOTH)


def dice_bce_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Dice loss + binary cross-entropy, per sample, shape ``(batch,)``.

    Combines a region-overlap term (robust to class imbalance between leaf and
    background) with a pixel-wise term for sharper boundaries.
    """
    dice_loss = 1.0 - dice_coefficient(y_true, y_pred)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)  # (batch, H, W)
    bce = tf.reduce_mean(bce, axis=[1, 2])                      # (batch,)
    return dice_loss + bce


# Friendly names so the metrics show up nicely in Keras logs / history.
dice_coefficient.__name__ = "dice"
iou_score.__name__ = "iou"

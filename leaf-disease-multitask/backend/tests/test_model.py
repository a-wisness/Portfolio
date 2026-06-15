"""Model-wiring tests. Need TensorFlow; skipped if not installed.

Build with ``encoder_weights=None`` so nothing is downloaded.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tensorflow")

import tensorflow as tf  # noqa: E402

from app.model import ENCODER_NAME, build_multitask_model, set_encoder_trainable  # noqa: E402

INPUT = (96, 96, 3)


@pytest.fixture(scope="module")
def model():
    return build_multitask_model(num_classes=5, input_shape=INPUT, encoder_weights=None)


def test_outputs_named_and_shaped(model):
    x = np.zeros((2, *INPUT), dtype=np.float32)
    out = model(x, training=False)
    assert set(out) == {"segmentation", "classification"}
    assert tuple(out["segmentation"].shape) == (2, INPUT[0], INPUT[1], 1)
    assert tuple(out["classification"].shape) == (2, 5)


def test_segmentation_in_unit_range(model):
    x = np.random.default_rng(0).random((1, *INPUT)).astype(np.float32) * 255
    seg = model(x, training=False)["segmentation"].numpy()
    assert seg.min() >= 0.0 and seg.max() <= 1.0  # sigmoid


def test_classification_is_softmax(model):
    x = np.random.default_rng(1).random((3, *INPUT)).astype(np.float32) * 255
    probs = model(x, training=False)["classification"].numpy()
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)


def test_encoder_freeze_unfreeze(model):
    set_encoder_trainable(model, False)
    encoder = model.get_layer(ENCODER_NAME)
    assert not encoder.trainable

    set_encoder_trainable(model, True, fine_tune_at=0)
    assert encoder.trainable
    # BatchNorm layers stay frozen to preserve running stats.
    bns = [l for l in encoder.layers if isinstance(l, tf.keras.layers.BatchNormalization)]
    assert bns and all(not l.trainable for l in bns)

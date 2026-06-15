"""Data-pipeline tests. Need TensorFlow; skipped if not installed."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tensorflow")

from app import data  # noqa: E402


def test_discover_classes(synthetic_settings):
    assert data.discover_classes(synthetic_settings) == [
        "alpha_leaf", "beta_leaf", "gamma_leaf"
    ]


def test_classification_sample_shapes_and_weights(synthetic_settings):
    ds = data.build_classification_dataset("train", synthetic_settings)
    image, targets, weights = next(iter(ds))
    h, w = synthetic_settings.image_size
    assert tuple(image.shape) == (h, w, 3)
    assert image.numpy().max() > 1.0  # kept in [0, 255], not normalized
    assert tuple(targets["classification"].shape) == (3,)
    assert tuple(targets["segmentation"].shape) == (h, w, 1)
    # Classification image: trains only the classification head.
    assert float(weights["classification"]) == 1.0
    assert float(weights["segmentation"]) == 0.0
    # One-hot label.
    assert float(np.sum(targets["classification"])) == 1.0


def test_segmentation_sample_shapes_and_weights(synthetic_settings):
    ds = data.build_segmentation_dataset("train", synthetic_settings, num_classes=3)
    image, targets, weights = next(iter(ds))
    h, w = synthetic_settings.image_size
    mask = targets["segmentation"].numpy()
    assert mask.shape == (h, w, 1)
    # Mask is binarized to {0, 1} and the synthetic rectangle gives both.
    assert set(np.unique(mask)).issubset({0.0, 1.0})
    assert mask.max() == 1.0 and mask.min() == 0.0
    # Segmentation image: trains only the segmentation head.
    assert float(weights["segmentation"]) == 1.0
    assert float(weights["classification"]) == 0.0
    # Dummy class is all zeros.
    assert float(np.sum(targets["classification"])) == 0.0


def test_multitask_dataset_batches(synthetic_settings):
    ds = data.build_multitask_dataset("val", synthetic_settings)
    image, targets, weights = next(iter(ds))
    assert image.shape[0] <= synthetic_settings.batch_size
    assert set(targets) == {"segmentation", "classification"}
    assert set(weights) == {"segmentation", "classification"}
    # Each sample is exactly one task: exactly one head has a non-zero weight
    # (the classification weight may differ from 1.0 due to class balancing).
    seg_w = weights["segmentation"].numpy()
    cls_w = weights["classification"].numpy()
    both_active = (seg_w > 0) & (cls_w > 0)
    neither_active = (seg_w == 0) & (cls_w == 0)
    assert not both_active.any()
    assert not neither_active.any()


def test_train_val_split_disjoint(synthetic_settings):
    pairs = data._list_segmentation_pairs(synthetic_settings)
    train, val = data._split(pairs, synthetic_settings.val_split, synthetic_settings.seed)
    assert len(val) == int(len(pairs) * synthetic_settings.val_split)
    assert set(map(tuple, train)).isdisjoint(set(map(tuple, val)))

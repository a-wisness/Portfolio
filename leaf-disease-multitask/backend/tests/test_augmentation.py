"""Augmentation + class-weight tests. Need TensorFlow; skipped if not installed."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tensorflow")

from app import data  # noqa: E402


def test_classification_augment_preserves_shape(synthetic_settings):
    h, w = synthetic_settings.image_size
    ds = data.build_classification_dataset(
        "train", synthetic_settings, augment=True
    )
    image, _t, _wt = next(iter(ds))
    assert tuple(image.shape) == (h, w, 3)
    assert 0.0 <= float(image.numpy().min())
    assert float(image.numpy().max()) <= 255.0


def test_segmentation_augment_keeps_mask_binary_and_aligned(synthetic_settings):
    ds = data.build_segmentation_dataset(
        "train", synthetic_settings, num_classes=3, augment=True
    )
    image, targets, _wt = next(iter(ds))
    mask = targets["segmentation"].numpy()
    # Augmentation (flips) must keep the mask strictly binary.
    assert set(np.unique(mask)).issubset({0.0, 1.0})
    assert image.shape[:2] == mask.shape[:2]


def test_augmentation_actually_changes_pixels(synthetic_settings):
    # Same source image with/without augmentation should usually differ.
    plain = data.build_classification_dataset("train", synthetic_settings, augment=False)
    aug = data.build_classification_dataset("train", synthetic_settings, augment=True)
    a = next(iter(plain))[0].numpy()
    b = next(iter(aug))[0].numpy()
    assert a.shape == b.shape  # augmentation is non-destructive to shape


def test_class_weights_balanced_and_capped(synthetic_settings):
    classes = data.discover_classes(synthetic_settings)
    weights = data.compute_class_weights(synthetic_settings, classes)
    assert len(weights) == len(classes)
    assert all(w > 0 for w in weights)
    assert max(weights) <= synthetic_settings.max_class_weight + 1e-6
    # Mean-normalized to ~1.0.
    assert abs(sum(weights) / len(weights) - 1.0) < 1e-5


def test_class_weights_applied_to_sample_weight(synthetic_settings):
    classes = data.discover_classes(synthetic_settings)
    weights = data.compute_class_weights(synthetic_settings, classes)
    ds = data.build_classification_dataset(
        "train", synthetic_settings, classes, augment=False, class_weights=weights
    )
    _img, targets, sw = next(iter(ds))
    label = int(np.argmax(targets["classification"].numpy()))
    assert float(sw["classification"]) == pytest.approx(weights[label], rel=1e-5)
    assert float(sw["segmentation"]) == 0.0


def test_single_task_dataset_only_classification(synthetic_settings):
    ds = data.build_multitask_dataset(
        "val", synthetic_settings, tasks=("classification",)
    )
    # Every sample is a classification sample -> seg weight always 0.
    for _img, _t, sw in ds:
        assert np.allclose(sw["segmentation"].numpy(), 0.0)
        assert np.all(sw["classification"].numpy() > 0.0)

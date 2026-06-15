"""Shared fixtures. Builds tiny synthetic datasets so tests run fully offline."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.config import Settings

CLASSES = ["alpha_leaf", "beta_leaf", "gamma_leaf"]
IMG = 64  # small images keep tests fast


def _write_rgb(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(IMG, IMG, 3), dtype=np.uint8)
    Image.fromarray(arr, "RGB").save(path)


def _write_mask(path: Path, seed: int) -> None:
    # A filled rectangle of value 1 (leaf) on a 0 background, mode 'P' with a
    # palette — mirrors the real dataset's quirky mask encoding.
    rng = np.random.default_rng(seed)
    m = np.zeros((IMG, IMG), dtype=np.uint8)
    y0, x0 = rng.integers(4, 20, size=2)
    m[y0:y0 + 30, x0:x0 + 30] = 1
    img = Image.fromarray(m, "P")
    img.putpalette([0, 0, 0, 38, 38, 38] + [0] * (256 * 3 - 6))
    img.save(path)


@pytest.fixture
def synthetic_settings(tmp_path: Path) -> Settings:
    """A Settings pointing at freshly built synthetic data + a temp artifacts dir."""
    cls_root = tmp_path / "Classification Data"
    seg_root = tmp_path / "Image Segmentation Data"

    # Classification: train/ and test/, 3 classes, 6 imgs each.
    for split in ("train", "test"):
        for ci, cls in enumerate(CLASSES):
            d = cls_root / split / cls
            d.mkdir(parents=True)
            for i in range(6):
                _write_rgb(d / f"{cls}_{i}.png", seed=100 * ci + i + (0 if split == "train" else 50))

    # Segmentation: data/{images,masks}, 10 pairs.
    img_dir = seg_root / "data" / "images"
    mask_dir = seg_root / "data" / "masks"
    img_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    for i in range(10):
        _write_rgb(img_dir / f"{i:05d}.jpg", seed=1000 + i)
        _write_mask(mask_dir / f"{i:05d}.png", seed=2000 + i)

    return Settings(
        classification_dir=cls_root,
        segmentation_dir=seg_root,
        artifacts_dir=tmp_path / "artifacts",
        img_height=IMG,
        img_width=IMG,
        batch_size=4,
        val_split=0.25,
        head_epochs=1,
        finetune_epochs=1,
    )

"""Input pipelines for the two disjoint datasets, unified for multi-task training.

The trick that makes one model train on both datasets: every sample is shaped
identically as ``(image, targets, sample_weights)`` where ``targets`` and
``sample_weights`` are dicts keyed by the model's two output names. A
classification image carries a real class label, a *dummy* zero mask, and weights
``seg=0, cls=1``; a segmentation image carries a real mask, a *dummy* zero class,
and weights ``seg=1, cls=0``. Keras then trains each head only on the samples
that actually labelled it, while the shared encoder sees every image.
"""
from __future__ import annotations

import random
from pathlib import Path

import tensorflow as tf

from .config import Settings, get_settings

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


# --------------------------------------------------------------------------
# Class discovery + file listing (eager, deterministic)
# --------------------------------------------------------------------------
def discover_classes(settings: Settings | None = None) -> list[str]:
    """Sorted class names from the classification ``train/`` subfolders."""
    settings = settings or get_settings()
    train_dir = Path(settings.classification_dir) / "train"
    classes = sorted(
        p.name for p in train_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    if not classes:
        raise FileNotFoundError(f"No class subfolders found under {train_dir}")
    return classes


def _list_classification_files(
    settings: Settings, classes: list[str]
) -> tuple[list[str], list[int]]:
    train_dir = Path(settings.classification_dir) / "train"
    paths: list[str] = []
    labels: list[int] = []
    for idx, cls in enumerate(classes):
        for p in sorted((train_dir / cls).iterdir()):
            if p.suffix in _IMG_EXTS:
                paths.append(str(p))
                labels.append(idx)
    return paths, labels


def _list_segmentation_pairs(settings: Settings) -> list[tuple[str, str]]:
    root = Path(settings.segmentation_dir)
    pairs: list[tuple[str, str]] = []
    for sub in ("data", "aug_data"):
        img_dir, mask_dir = root / sub / "images", root / sub / "masks"
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix not in _IMG_EXTS:
                continue
            mask = mask_dir / (img.stem + ".png")
            if mask.exists():
                pairs.append((str(img), str(mask)))
    return pairs


def _split(items: list, val_split: float, seed: int) -> tuple[list, list]:
    """Deterministic train/val split."""
    items = list(items)
    random.Random(seed).shuffle(items)
    n_val = int(len(items) * val_split)
    return items[n_val:], items[:n_val]


# --------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------
def _load_image(path: tf.Tensor, size: tuple[int, int]) -> tf.Tensor:
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, size, method="bilinear")
    img.set_shape([size[0], size[1], 3])
    return tf.cast(img, tf.float32)  # kept in [0, 255]; model rescales internally


def _load_mask(path: tf.Tensor, size: tuple[int, int]) -> tf.Tensor:
    raw = tf.io.read_file(path)
    # Decode as 3ch so paletted/RGB/grayscale masks all behave; collapse to a
    # single channel, resize nearest to stay binary, then threshold > 0.
    m = tf.io.decode_png(raw, channels=3)
    m = tf.reduce_max(m, axis=-1, keepdims=True)
    m = tf.image.resize(m, size, method="nearest")
    m.set_shape([size[0], size[1], 1])
    return tf.cast(m > 0, tf.float32)


# --------------------------------------------------------------------------
# Augmentation (task-correct)
# --------------------------------------------------------------------------
def _color_jitter(image: tf.Tensor) -> tf.Tensor:
    """Photometric jitter on an image in [0, 255] (leaves RGB geometry intact)."""
    image = tf.image.random_brightness(image, max_delta=0.1 * 255)
    image = tf.image.random_contrast(image, 0.85, 1.15)
    image = tf.image.random_saturation(image, 0.85, 1.15)
    return tf.clip_by_value(image, 0.0, 255.0)


def _augment_classification(image: tf.Tensor) -> tf.Tensor:
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    return _color_jitter(image)


def _augment_segmentation(image: tf.Tensor, mask: tf.Tensor):
    """Geometric transforms must hit image AND mask identically; concatenating
    them on the channel axis guarantees the same random flips are applied to
    both. Color jitter then touches the image only."""
    combined = tf.concat([image, mask], axis=-1)              # (H, W, 4)
    combined = tf.image.random_flip_left_right(combined)
    combined = tf.image.random_flip_up_down(combined)
    image, mask = combined[..., :3], combined[..., 3:]
    image = _color_jitter(image)
    return image, mask


# --------------------------------------------------------------------------
# Class weights (imbalance handling)
# --------------------------------------------------------------------------
def compute_class_weights(
    settings: Settings, classes: list[str]
) -> list[float]:
    """Balanced inverse-frequency weights, capped, mean-normalized to ~1.0.

    One class (``Tomato two spotted spider mites leaf``) has only 2 training
    images; without this it is effectively ignored."""
    paths, labels = _list_classification_files(settings, classes)
    train_pairs, _ = _split(list(zip(paths, labels)), settings.val_split, settings.seed)
    counts = [0] * len(classes)
    for _, lbl in train_pairs:
        counts[lbl] += 1
    total = sum(counts)
    n = len(classes)
    weights = []
    for c in counts:
        w = total / (n * c) if c > 0 else settings.max_class_weight
        weights.append(min(w, settings.max_class_weight))
    # Normalize so the average weight is ~1 (keeps loss scale stable).
    mean_w = sum(weights) / len(weights)
    return [w / mean_w for w in weights]


# --------------------------------------------------------------------------
# Per-task datasets (each yields the unified multi-task structure)
# --------------------------------------------------------------------------
def build_classification_dataset(
    split: str,
    settings: Settings | None = None,
    classes: list[str] | None = None,
    augment: bool | None = None,
    class_weights: list[float] | None = None,
) -> tf.data.Dataset:
    settings = settings or get_settings()
    classes = classes or discover_classes(settings)
    num_classes = len(classes)
    size = settings.image_size
    augment = settings.augment if augment is None else augment
    do_augment = augment and split == "train"
    cw = tf.constant(class_weights, tf.float32) if class_weights else None

    paths, labels = _list_classification_files(settings, classes)
    pairs = list(zip(paths, labels))
    train_pairs, val_pairs = _split(pairs, settings.val_split, settings.seed)
    chosen = train_pairs if split == "train" else val_pairs
    if not chosen:
        raise ValueError(f"No classification samples for split={split!r}")
    sel_paths = [p for p, _ in chosen]
    sel_labels = [lbl for _, lbl in chosen]

    ds = tf.data.Dataset.from_tensor_slices((sel_paths, sel_labels))

    def _map(path, label):
        image = _load_image(path, size)
        if do_augment:
            image = _augment_classification(image)
        cls = tf.one_hot(label, num_classes)
        seg = tf.zeros([size[0], size[1], 1], tf.float32)        # dummy mask
        cls_weight = cw[label] if cw is not None else tf.constant(1.0)
        targets = {"segmentation": seg, "classification": cls}
        weights = {
            "segmentation": tf.constant(0.0),
            "classification": cls_weight,
        }
        return image, targets, weights

    return ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)


def build_segmentation_dataset(
    split: str,
    settings: Settings | None = None,
    num_classes: int | None = None,
    augment: bool | None = None,
) -> tf.data.Dataset:
    settings = settings or get_settings()
    if num_classes is None:
        num_classes = len(discover_classes(settings))
    size = settings.image_size
    augment = settings.augment if augment is None else augment
    do_augment = augment and split == "train"

    pairs = _list_segmentation_pairs(settings)
    train_pairs, val_pairs = _split(pairs, settings.val_split, settings.seed)
    chosen = train_pairs if split == "train" else val_pairs
    if not chosen:
        raise ValueError(f"No segmentation samples for split={split!r}")
    img_paths = [i for i, _ in chosen]
    mask_paths = [m for _, m in chosen]

    ds = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))

    def _map(img_path, mask_path):
        image = _load_image(img_path, size)
        seg = _load_mask(mask_path, size)
        if do_augment:
            image, seg = _augment_segmentation(image, seg)
        cls = tf.zeros([num_classes], tf.float32)                # dummy class
        targets = {"segmentation": seg, "classification": cls}
        weights = {
            "segmentation": tf.constant(1.0),
            "classification": tf.constant(0.0),
        }
        return image, targets, weights

    return ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)


# --------------------------------------------------------------------------
# Unified multi-task dataset
# --------------------------------------------------------------------------
def build_multitask_dataset(
    split: str,
    settings: Settings | None = None,
    classes: list[str] | None = None,
    tasks: tuple[str, ...] = ("segmentation", "classification"),
) -> tf.data.Dataset:
    """Interleave the selected tasks into one batched, prefetched dataset.

    ``tasks`` selects which datasets feed the stream — used by the ablation
    harness to train a single-task baseline against the multi-task model.
    """
    settings = settings or get_settings()
    classes = classes or discover_classes(settings)
    num_classes = len(classes)

    class_weights = None
    if settings.use_class_weights:
        class_weights = compute_class_weights(settings, classes)

    sources, mix = [], []
    if "segmentation" in tasks:
        sources.append(build_segmentation_dataset(split, settings, num_classes))
        mix.append(settings.seg_sample_ratio)
    if "classification" in tasks:
        sources.append(
            build_classification_dataset(split, settings, classes,
                                          class_weights=class_weights)
        )
        mix.append(1.0 - settings.seg_sample_ratio)
    if not sources:
        raise ValueError("tasks must include 'segmentation' and/or 'classification'")

    if split == "train":
        sources = [s.shuffle(512, seed=settings.seed).repeat() for s in sources]
        if len(sources) == 1:
            ds = sources[0]
        else:
            ds = tf.data.Dataset.sample_from_datasets(
                sources, weights=mix, seed=settings.seed
            )
    else:
        # Validation: cover every sample once (no repeat/sampling).
        ds = sources[0]
        for extra in sources[1:]:
            ds = ds.concatenate(extra)

    return ds.batch(settings.batch_size).prefetch(tf.data.AUTOTUNE)


def steps_per_epoch(
    split: str,
    settings: Settings | None = None,
    tasks: tuple[str, ...] = ("segmentation", "classification"),
) -> int:
    """Number of batches for one pass over the selected training data (the
    repeated multi-task stream needs an explicit step count)."""
    settings = settings or get_settings()
    classes = discover_classes(settings)
    total = 0
    if "segmentation" in tasks:
        total += len(build_segmentation_dataset(split, settings, len(classes)))
    if "classification" in tasks:
        total += len(build_classification_dataset(split, settings, classes))
    steps = max(1, total // settings.batch_size)
    if settings.max_steps_per_epoch:
        steps = min(steps, settings.max_steps_per_epoch)
    return steps

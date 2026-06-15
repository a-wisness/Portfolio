"""Held-out evaluation of a trained artifact.

Reports segmentation IoU/Dice on the segmentation validation split and a
classification report (accuracy + per-class precision/recall) on the *separate*
``Classification Data/test`` folder — a genuine held-out set, fixing the original
notebook's train/test leakage.

Run:  ``python -m app.evaluate``
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from .config import Settings, get_settings
from .data import _IMG_EXTS, _load_image, build_segmentation_dataset, discover_classes
from .losses import dice_bce_loss, dice_coefficient, iou_score
from .registry import Registry


def _load_trained_model(settings: Settings):
    files = Registry(settings).resolve_active()
    if files is None:
        raise FileNotFoundError("No trained model registered.")
    return tf.keras.models.load_model(
        files["model"],
        custom_objects={
            "dice_bce_loss": dice_bce_loss,
            "iou": iou_score,
            "iou_score": iou_score,
            "dice": dice_coefficient,
            "dice_coefficient": dice_coefficient,
        },
        compile=False,
    )


def evaluate_segmentation(model, settings: Settings) -> dict:
    ds = build_segmentation_dataset("val", settings).batch(settings.batch_size)
    ious, dices = [], []
    for image, targets, _weights in ds:
        pred = model(image, training=False)["segmentation"]
        ious.append(iou_score(targets["segmentation"], pred).numpy())
        dices.append(dice_coefficient(targets["segmentation"], pred).numpy())
    return {
        "iou": float(np.concatenate(ious).mean()),
        "dice": float(np.concatenate(dices).mean()),
        "num_samples": int(sum(len(x) for x in ious)),
    }


def evaluate_classification(model, settings: Settings, classes: list[str]) -> dict:
    test_dir = Path(settings.classification_dir) / "test"
    size = settings.image_size
    y_true, y_pred = [], []
    for idx, cls in enumerate(classes):
        cls_dir = test_dir / cls
        if not cls_dir.is_dir():
            continue
        for p in sorted(cls_dir.iterdir()):
            if p.suffix not in _IMG_EXTS:
                continue
            img = _load_image(tf.constant(str(p)), size)[None, ...]
            probs = model(img, training=False)["classification"].numpy()[0]
            y_true.append(idx)
            y_pred.append(int(np.argmax(probs)))

    y_true_a, y_pred_a = np.array(y_true), np.array(y_pred)
    accuracy = float((y_true_a == y_pred_a).mean()) if len(y_true_a) else 0.0
    result = {"accuracy": accuracy, "num_samples": int(len(y_true_a))}
    try:
        from sklearn.metrics import classification_report, confusion_matrix

        present = sorted(set(y_true) | set(y_pred))
        result["report"] = classification_report(
            y_true_a, y_pred_a,
            labels=present,
            target_names=[classes[i] for i in present],
            zero_division=0,
            output_dict=True,
        )
        result["confusion_matrix"] = confusion_matrix(
            y_true_a, y_pred_a, labels=list(range(len(classes)))
        ).tolist()
    except ImportError:
        result["report"] = None
        result["confusion_matrix"] = None
    return result


def evaluate(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    files = Registry(settings).resolve_active()
    if files is None:
        raise FileNotFoundError("No trained model registered.")
    classes = json.loads(files["labels"].read_text())
    model = _load_trained_model(settings)

    seg = evaluate_segmentation(model, settings)
    cls = evaluate_classification(model, settings, classes)
    cls["labels"] = classes  # for confusion-matrix axes in reporting
    summary = {"segmentation": seg, "classification": {
        "accuracy": cls["accuracy"], "num_samples": cls["num_samples"]}}

    print("Segmentation (val):", {k: round(v, 4) if isinstance(v, float) else v
                                   for k, v in seg.items()})
    print(f"Classification (held-out test): accuracy={cls['accuracy']:.4f} "
          f"over {cls['num_samples']} images")
    return {"segmentation": seg, "classification": cls, "summary": summary}


if __name__ == "__main__":
    evaluate()

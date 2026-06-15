"""Two-phase multi-task training -> a single serialized artifact.

Phase A trains the two heads with the shared encoder frozen; Phase B fine-tunes
the top encoder blocks at a low learning rate. The best model (by validation
loss) plus the class label map and a metadata sidecar are written to
``artifacts/``; training curves + a markdown report are written to ``reports/``.

Run:  ``python -m app.train``
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import tensorflow as tf

from . import __version__
from .config import Settings, get_settings
from .data import (
    build_multitask_dataset,
    compute_class_weights,
    discover_classes,
    steps_per_epoch,
)
from .losses import dice_bce_loss, dice_coefficient, iou_score
from .model import build_multitask_model, set_encoder_trainable
from .registry import Registry

Tasks = tuple[str, ...]
BOTH: Tasks = ("segmentation", "classification")


def _compile(model: tf.keras.Model, lr: float, settings: Settings, tasks: Tasks) -> None:
    # Zero out the loss weight of any head not being trained (ablation support).
    seg_w = settings.seg_loss_weight if "segmentation" in tasks else 0.0
    cls_w = settings.cls_loss_weight if "classification" in tasks else 0.0
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss={
            "segmentation": dice_bce_loss,
            "classification": tf.keras.losses.CategoricalCrossentropy(),
        },
        loss_weights={"segmentation": seg_w, "classification": cls_w},
        weighted_metrics={
            "segmentation": [iou_score, dice_coefficient],
            "classification": ["accuracy"],
        },
    )


def run_training(
    settings: Settings,
    classes: list[str],
    tasks: Tasks = BOTH,
    callbacks: list | None = None,
) -> tuple[tf.keras.Model, "tf.keras.callbacks.History", "tf.keras.callbacks.History"]:
    """Build + two-phase fit. Returns (model, phase_a_history, phase_b_history)."""
    tf.keras.utils.set_random_seed(settings.seed)
    train_ds = build_multitask_dataset("train", settings, classes, tasks=tasks)
    val_ds = build_multitask_dataset("val", settings, classes, tasks=tasks)
    train_steps = steps_per_epoch("train", settings, tasks=tasks)

    model = build_multitask_model(
        num_classes=len(classes),
        input_shape=settings.input_shape,
        dropout=settings.dropout,
    )

    # --- Phase A: frozen encoder, train the heads ------------------------
    print("\n=== Phase A: training heads (encoder frozen) ===")
    set_encoder_trainable(model, False)
    _compile(model, settings.head_lr, settings, tasks)
    history_a = model.fit(
        train_ds, validation_data=val_ds,
        epochs=settings.head_epochs, steps_per_epoch=train_steps,
        callbacks=callbacks or [],
    )

    # --- Phase B: fine-tune the top encoder blocks -----------------------
    print("\n=== Phase B: fine-tuning encoder ===")
    set_encoder_trainable(model, True, fine_tune_at=settings.finetune_at)
    _compile(model, settings.finetune_lr, settings, tasks)
    total_epochs = settings.head_epochs + settings.finetune_epochs
    history_b = model.fit(
        train_ds, validation_data=val_ds,
        epochs=total_epochs, initial_epoch=len(history_a.epoch),
        steps_per_epoch=train_steps,
        callbacks=callbacks or [],
    )
    return model, history_a, history_b


def train(settings: Settings | None = None, write_reports: bool = True) -> Path:
    """Train the multi-task model end to end and return the artifact directory."""
    settings = settings or get_settings()
    classes = discover_classes(settings)
    num_classes = len(classes)
    print(f"Discovered {num_classes} classes.")
    if settings.use_class_weights:
        cw = compute_class_weights(settings, classes)
        print(f"Class weights: min={min(cw):.2f} max={max(cw):.2f} "
              f"(balanced, capped at {settings.max_class_weight}).")

    artifacts_dir = Path(settings.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Register a fresh version up front so the checkpoint writes straight into it.
    registry = Registry(settings)
    version_id, version_dir = registry.create_version()
    model_file = version_dir / "model.keras"
    print(f"Training model version: {version_id}")

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(model_file), monitor="val_loss",
        save_best_only=True, verbose=1,
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min",
        patience=settings.early_stopping_patience,
        restore_best_weights=True, verbose=1,
    )

    model, history_a, history_b = run_training(
        settings, classes, tasks=BOTH, callbacks=[checkpoint, early_stop]
    )

    # Save the (best-weights, restored in-memory) model into the version dir.
    model.save(model_file)

    val_ds = build_multitask_dataset("val", settings, classes, tasks=BOTH)
    val_metrics = model.evaluate(val_ds, return_dict=True, verbose=0)
    metadata = {
        "version": __version__,
        "version_id": version_id,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "num_classes": num_classes,
        "input_shape": list(settings.input_shape),
        "mask_threshold": settings.mask_threshold,
        "val_metrics": {k: float(v) for k, v in val_metrics.items()},
        "config": {
            "batch_size": settings.batch_size,
            "head_epochs": settings.head_epochs,
            "finetune_epochs": settings.finetune_epochs,
            "head_lr": settings.head_lr,
            "finetune_lr": settings.finetune_lr,
            "seg_sample_ratio": settings.seg_sample_ratio,
            "augment": settings.augment,
            "use_class_weights": settings.use_class_weights,
        },
        "history": {
            "phase_a": {k: [float(x) for x in v] for k, v in history_a.history.items()},
            "phase_b": {k: [float(x) for x in v] for k, v in history_b.history.items()},
        },
    }
    # Write the label map + metadata and make this version active.
    registry.finalize(version_id, classes, metadata, set_active=True)
    # Keep a flat metadata copy for the reporting step.
    settings.metadata_path.write_text(json.dumps(metadata, indent=2))

    print(f"\nSaved model version {version_id} (now active) to {version_dir}")
    print("Validation metrics:", {k: round(v, 4) for k, v in val_metrics.items()})

    if write_reports:
        from .evaluate import evaluate
        from .reporting import generate_reports

        reports_dir = artifacts_dir.parent / "reports"
        try:
            evaluation = evaluate(settings)
        except Exception as exc:  # reporting must never break a good training run
            print(f"(evaluation skipped: {exc})")
            evaluation = None
        out = generate_reports(settings.metadata_path, reports_dir, evaluation)
        print(f"Reports written to {reports_dir} ({out}).")

    return artifacts_dir


if __name__ == "__main__":
    train()

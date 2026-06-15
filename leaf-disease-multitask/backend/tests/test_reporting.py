"""Reporting tests — no TensorFlow required (pure dict -> files)."""
from __future__ import annotations

import json

from app import reporting

METADATA = {
    "version": "0.1.0",
    "created_utc": "2026-06-15T00:00:00+00:00",
    "num_classes": 3,
    "config": {"batch_size": 4, "augment": True},
    "val_metrics": {"loss": 1.1, "classification_accuracy": 0.5,
                    "segmentation_iou": 0.42},
    "history": {
        "phase_a": {
            "loss": [2.0, 1.6],
            "val_loss": [1.9, 1.7],
            "classification_accuracy": [0.2, 0.4],
            "val_classification_accuracy": [0.25, 0.35],
            "segmentation_iou": [0.1, 0.3],
            "val_segmentation_iou": [0.12, 0.28],
            "segmentation_dice": [0.2, 0.4],
            "val_segmentation_dice": [0.22, 0.38],
        },
        "phase_b": {
            "loss": [1.5, 1.3],
            "val_loss": [1.6, 1.55],
            "classification_accuracy": [0.45, 0.55],
            "val_classification_accuracy": [0.4, 0.5],
            "segmentation_iou": [0.35, 0.45],
            "val_segmentation_iou": [0.3, 0.42],
            "segmentation_dice": [0.45, 0.55],
            "val_segmentation_dice": [0.4, 0.52],
        },
    },
}

EVALUATION = {
    "segmentation": {"iou": 0.43, "dice": 0.55, "num_samples": 20},
    "classification": {"accuracy": 0.5, "num_samples": 18, "report": {
        "alpha_leaf": {"precision": 0.6, "recall": 0.5, "f1-score": 0.55, "support": 6},
    }},
}


def test_merge_history_concatenates_phases():
    merged = reporting._merge_history(METADATA["history"])
    assert merged["loss"] == [2.0, 1.6, 1.5, 1.3]
    assert len(merged["val_classification_accuracy"]) == 4


def test_write_markdown_report(tmp_path):
    out = tmp_path / "report.md"
    reporting.write_markdown_report(METADATA, EVALUATION, out, curves_filename="c.png")
    text = out.read_text()
    assert "# LeafLens — training report" in text
    assert "Held-out evaluation" in text
    assert "alpha_leaf" in text          # per-class table rendered
    assert "![training curves](c.png)" in text


def test_generate_reports(tmp_path):
    meta_path = tmp_path / "metadata.json"
    meta_path.write_text(json.dumps(METADATA))
    reports_dir = tmp_path / "reports"
    out = reporting.generate_reports(meta_path, reports_dir, EVALUATION)
    assert (reports_dir / "report.md").exists()
    # Curves render only if matplotlib is installed; either way report exists.
    assert isinstance(out["curves"], bool)
    if out["curves"]:
        assert (reports_dir / "training_curves.png").exists()

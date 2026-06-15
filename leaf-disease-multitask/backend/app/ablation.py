"""Single-task vs multi-task ablation.

Trains a classification-only baseline (the shared encoder sees *only* the
classification data) and the multi-task model (encoder also learns from the
segmentation data), then compares validation classification accuracy. The point
is to show whether the auxiliary segmentation task improves the disease
classifier — the motivation for unifying the two models.

Run:  ``python -m app.ablation``  (use a small epoch budget; this trains twice)
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import Settings, get_settings
from .data import build_multitask_dataset, discover_classes
from .train import run_training


def _val_classification_accuracy(model, settings: Settings, classes: list[str]) -> float:
    val_ds = build_multitask_dataset(
        "val", settings, classes, tasks=("classification",)
    )
    metrics = model.evaluate(val_ds, return_dict=True, verbose=0)
    # Key is "classification_accuracy" for the multi-output model.
    for key in ("classification_accuracy", "accuracy"):
        if key in metrics:
            return float(metrics[key])
    return float("nan")


def run_ablation(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    classes = discover_classes(settings)
    results = {}

    for name, tasks in [
        ("classification_only", ("classification",)),
        ("multitask", ("segmentation", "classification")),
    ]:
        print(f"\n########## Ablation arm: {name} ({tasks}) ##########")
        model, _, _ = run_training(settings, classes, tasks=tasks)
        acc = _val_classification_accuracy(model, settings, classes)
        results[name] = acc
        print(f"{name}: val classification accuracy = {acc:.4f}")

    delta = results["multitask"] - results["classification_only"]
    summary = {
        "arms": results,
        "multitask_minus_single": delta,
        "helps": delta > 0,
    }
    print("\n=== Ablation summary ===")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    settings = get_settings()
    summary = run_ablation(settings)
    reports_dir = Path(settings.artifacts_dir).parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "ablation.json").write_text(json.dumps(summary, indent=2))
    lines = [
        "# Ablation — single-task vs multi-task", "",
        "Validation classification accuracy:", "",
        "| arm | accuracy |", "|---|---|",
        f"| classification-only | {summary['arms']['classification_only']:.4f} |",
        f"| multi-task | {summary['arms']['multitask']:.4f} |",
        "",
        f"Multi-task − single-task: **{summary['multitask_minus_single']:+.4f}** "
        f"({'helps' if summary['helps'] else 'no improvement'}).",
        "",
    ]
    (reports_dir / "ablation.md").write_text("\n".join(lines))
    print(f"Ablation report written to {reports_dir}")


if __name__ == "__main__":
    main()

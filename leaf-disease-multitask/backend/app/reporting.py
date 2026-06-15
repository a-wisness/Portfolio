"""Training-curve plots + a markdown evaluation report.

``matplotlib`` is imported lazily (and the module degrades gracefully if it is
absent) so it stays a dev-only dependency and never blocks the API/training.
"""
from __future__ import annotations

import json
from pathlib import Path


def _merge_history(history: dict) -> dict[str, list[float]]:
    """Concatenate the per-phase history dicts written by training."""
    merged: dict[str, list[float]] = {}
    for phase in ("phase_a", "phase_b"):
        for key, values in history.get(phase, {}).items():
            merged.setdefault(key, []).extend(values)
    return merged


def plot_training_curves(history: dict, out_path: Path) -> bool:
    """Render loss + per-task metric curves to ``out_path`` (PNG).

    Returns False (without raising) if matplotlib is unavailable.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    h = _merge_history(history)
    epochs = range(1, len(h.get("loss", [])) + 1)

    panels = [
        ("loss", ["loss", "val_loss"], "Total loss"),
        ("classification_accuracy",
         ["classification_accuracy", "val_classification_accuracy"],
         "Classification accuracy"),
        ("segmentation_iou",
         ["segmentation_iou", "val_segmentation_iou"], "Segmentation IoU"),
        ("segmentation_dice",
         ["segmentation_dice", "val_segmentation_dice"], "Segmentation Dice"),
    ]
    available = [p for p in panels if p[1][0] in h]
    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4))
    if len(available) == 1:
        axes = [axes]

    for ax, (_key, series, title) in zip(axes, available):
        for s in series:
            if s in h:
                ax.plot(epochs, h[s], label="val" if s.startswith("val") else "train")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return True


def plot_confusion_matrix(matrix, labels, out_path: Path) -> bool:
    """Render a confusion matrix heatmap. Returns False if matplotlib is absent."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False

    cm = np.asarray(matrix)
    n = cm.shape[0]
    fig, ax = plt.subplots(figsize=(max(6, n * 0.4), max(5, n * 0.4)))
    im = ax.imshow(cm, cmap="viridis")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix (held-out test)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def write_markdown_report(
    metadata: dict, evaluation: dict | None, out_path: Path,
    curves_filename: str | None = None,
    confusion_filename: str | None = None,
) -> None:
    """Write a human-readable evaluation report to ``out_path``."""
    lines: list[str] = ["# LeafLens — training report", ""]
    lines.append(f"- **Model version:** {metadata.get('version', '?')}")
    lines.append(f"- **Created (UTC):** {metadata.get('created_utc', '?')}")
    lines.append(f"- **Classes:** {metadata.get('num_classes', '?')}")
    cfg = metadata.get("config", {})
    if cfg:
        lines.append("- **Config:** " + ", ".join(f"{k}={v}" for k, v in cfg.items()))
    lines.append("")

    val = metadata.get("val_metrics", {})
    if val:
        lines += ["## Validation metrics (mixed multi-task stream)", "",
                  "| metric | value |", "|---|---|"]
        for k in sorted(val):
            lines.append(f"| {k} | {val[k]:.4f} |")
        lines.append("")

    if evaluation:
        seg = evaluation.get("segmentation", {})
        cls = evaluation.get("classification", {})
        lines += ["## Held-out evaluation", ""]
        if seg:
            lines.append(f"**Segmentation (val split):** IoU "
                         f"{seg.get('iou', float('nan')):.4f}, Dice "
                         f"{seg.get('dice', float('nan')):.4f} "
                         f"over {seg.get('num_samples', '?')} images.")
        if cls:
            lines.append(f"**Classification (held-out test):** accuracy "
                         f"{cls.get('accuracy', float('nan')):.4f} over "
                         f"{cls.get('num_samples', '?')} images.")
        lines.append("")
        report = cls.get("report") if cls else None
        if isinstance(report, dict):
            lines += ["### Per-class (precision / recall / f1)", "",
                      "| class | precision | recall | f1 | support |",
                      "|---|---|---|---|---|"]
            for name, m in report.items():
                if not isinstance(m, dict) or "precision" not in m:
                    continue
                lines.append(
                    f"| {name} | {m['precision']:.2f} | {m['recall']:.2f} | "
                    f"{m['f1-score']:.2f} | {int(m['support'])} |"
                )
            lines.append("")

    if curves_filename:
        lines += ["## Training curves", "", f"![training curves]({curves_filename})", ""]
    if confusion_filename:
        lines += ["## Confusion matrix", "",
                  f"![confusion matrix]({confusion_filename})", ""]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def generate_reports(
    metadata_path: Path, reports_dir: Path, evaluation: dict | None = None
) -> dict:
    """Read a metadata sidecar and emit curves + a markdown report."""
    metadata = json.loads(Path(metadata_path).read_text())
    reports_dir = Path(reports_dir)
    curves_name = "training_curves.png"
    has_curves = plot_training_curves(metadata.get("history", {}),
                                      reports_dir / curves_name)

    # Confusion matrix (only if evaluation provided one + labels).
    confusion_name = "confusion_matrix.png"
    has_confusion = False
    cls = (evaluation or {}).get("classification", {}) if evaluation else {}
    cm, labels = cls.get("confusion_matrix"), cls.get("labels")
    if cm and labels:
        has_confusion = plot_confusion_matrix(cm, labels, reports_dir / confusion_name)

    write_markdown_report(
        metadata, evaluation, reports_dir / "report.md",
        curves_filename=curves_name if has_curves else None,
        confusion_filename=confusion_name if has_confusion else None,
    )
    return {
        "curves": has_curves,
        "confusion_matrix": has_confusion,
        "report": str(reports_dir / "report.md"),
    }

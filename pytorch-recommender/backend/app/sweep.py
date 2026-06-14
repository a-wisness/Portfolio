"""Hyperparameter sweep over NeuMF, evaluated with the leave-one-out protocol.

Run:  python -m app.sweep [epochs]

Trains NeuMF for each combination in a small grid (embedding dim × negatives ×
learning rate), reusing one preprocessed dataset, and writes a table sorted by
NDCG@K to reports/sweep.md. The grid is intentionally small so the run finishes
in minutes; widen DEFAULT_GRID for a real search.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

from .config import settings
from .data import load_and_prepare
from .training import set_seed, train_model

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"

DEFAULT_GRID: dict[str, list] = {
    "gmf_dim": [16, 32],
    "num_negatives": [4, 8],
    "learning_rate": [1e-3],
}


def run_sweep(epochs: int, grid: dict[str, list] | None = None) -> list[dict]:
    grid = grid or DEFAULT_GRID
    set_seed(settings.seed)
    print(f"Loading dataset: {settings.dataset}")
    data = load_and_prepare(settings)  # dataset is independent of the swept knobs
    print(f"Users: {data.num_users}  Items: {data.num_items}\n")

    keys = list(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))
    k = settings.top_k
    results: list[dict] = []

    for combo in combos:
        overrides = dict(zip(keys, combo))
        cfg = settings.model_copy(update=overrides)
        set_seed(cfg.seed)  # same seed per run -> differences are from the knobs
        print(f"Training {overrides} ...")
        _model, _history, metrics = train_model("neumf", data, cfg, "bce", epochs=epochs)
        results.append({**overrides, **metrics})
        print(f"  -> HR@{k}={metrics[f'hr@{k}']:.4f}  NDCG@{k}={metrics[f'ndcg@{k}']:.4f}\n")

    write_report(results, keys, epochs)
    return results


def write_report(results: list[dict], keys: list[str], epochs: int) -> Path:
    k = settings.top_k
    ranked = sorted(results, key=lambda r: r[f"ndcg@{k}"], reverse=True)

    header = "| " + " | ".join(keys) + f" | HR@{k} | NDCG@{k} |"
    divider = "|" + "---|" * len(keys) + "---:|---:|"
    lines = [
        f"# Hyperparameter Sweep — {settings.dataset}",
        "",
        f"NeuMF, {epochs} epochs per run, leave-one-out evaluation. Best NDCG@{k} marked ★.",
        "",
        header,
        divider,
    ]
    for i, r in enumerate(ranked):
        cells = " | ".join(str(r[key]) for key in keys)
        star = " ★" if i == 0 else ""
        lines.append(f"| {cells}{star} | {r[f'hr@{k}']:.4f} | {r[f'ndcg@{k}']:.4f} |")
    lines.append("")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "sweep.md"
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else max(8, settings.epochs // 2)
    run_sweep(epochs)
    print(f"Wrote {REPORTS_DIR / 'sweep.md'}")


if __name__ == "__main__":
    main()

"""Training and evaluation entry point for the served model (NeuMF).

Run:  python -m app.train

Downloads MovieLens (if needed), trains NeuMF on implicit feedback with
per-epoch negative sampling, evaluates with the leave-one-out HR@K / NDCG@K
protocol each epoch, and serializes the best model to ``artifacts/model.pt``.

The actual training loop lives in ``training.py`` (shared with the benchmark);
this module owns the CLI and artifact serialization.
"""

from __future__ import annotations

from datetime import datetime, timezone

from . import registry
from .config import settings
from .data import PreparedData, load_and_prepare
from .model import NeuMF
from .training import set_seed, train_model


def build_artifact(
    model: NeuMF, data: PreparedData, metrics: dict[str, float], version: str, created_at: str
) -> dict:
    return {
        "version": version,
        "created_at": created_at,
        "config": {
            "num_users": data.num_users,
            "num_items": data.num_items,
            "gmf_dim": settings.gmf_dim,
            "mlp_layers": list(settings.mlp_layers),
            "dropout": settings.dropout,
            "dataset": settings.dataset,
            "min_positive_rating": settings.min_positive_rating,
        },
        "state_dict": model.state_dict(),
        "user_id_to_idx": data.user_id_to_idx,
        "item_id_to_idx": data.item_id_to_idx,
        "movies": data.movies,
        "user_seen": {u: sorted(items) for u, items in data.user_seen.items()},
        "metrics": metrics,
    }


def train() -> str:
    set_seed(settings.seed)
    print(f"Loading dataset: {settings.dataset}")
    data = load_and_prepare(settings)
    print(
        f"Users: {data.num_users}  Items: {data.num_items}  "
        f"Train positives: {len(data.train_pairs)}  Test users: {len(data.test_items)}"
    )

    model, _history, metrics = train_model("neumf", data, settings, objective="bce", verbose=True)

    version = registry.new_version_id()
    created_at = datetime.now(timezone.utc).isoformat()
    bundle = build_artifact(model, data, metrics, version, created_at)
    meta = {
        "version": version,
        "created_at": created_at,
        "dataset": settings.dataset,
        "num_users": data.num_users,
        "num_items": data.num_items,
        "metrics": metrics,
    }
    registry.save_version(bundle, meta)

    k = settings.top_k
    print(
        f"Best HR@{k}={metrics[f'hr@{k}']:.4f}  NDCG@{k}={metrics[f'ndcg@{k}']:.4f}. "
        f"Saved + activated version {version}."
    )
    return version


if __name__ == "__main__":
    train()

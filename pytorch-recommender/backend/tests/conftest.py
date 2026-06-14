"""Shared test fixtures.

The suite runs fully offline: no MovieLens download and no real training. API
tests load a tiny artifact built from a freshly-initialized NeuMF, so the real
inference paths (model load, scoring, similarity) run against a real — if
untrained — model.
"""

from __future__ import annotations

import pytest
import numpy as np
import torch

from app import recommender, registry
from app.config import settings
from app.data import PreparedData
from app.model import NeuMF
from app.observability import collector


def make_tiny_data() -> PreparedData:
    """A small but valid PreparedData for training/evaluation tests."""
    num_users, num_items = 6, 12
    train_pairs, test_items, eval_negatives, user_seen, user_sequences = [], {}, {}, {}, {}
    for u in range(num_users):
        items = list(dict.fromkeys([(u * 2) % num_items, (u * 2 + 1) % num_items, (u + 3) % num_items]))
        while len(items) < 3:  # guarantee 3 distinct items per user
            items.append((items[-1] + 1) % num_items)
            items = list(dict.fromkeys(items))
        seen = set(items)
        user_seen[u] = seen
        for it in items[:2]:
            train_pairs.append((u, it))
        user_sequences[u] = list(items[:2])  # ordered training items
        test_items[u] = items[2]
        eval_negatives[u] = [i for i in range(num_items) if i not in seen][:5]
    movies = {
        i: {"movie_id": 1000 + i, "title": f"Movie {i}", "genres": ["Drama"]}
        for i in range(num_items)
    }
    return PreparedData(
        num_users=num_users,
        num_items=num_items,
        train_pairs=np.array(train_pairs, dtype=np.int64),
        user_seen=user_seen,
        test_items=test_items,
        eval_negatives=eval_negatives,
        movies=movies,
        user_id_to_idx={100 + u: u for u in range(num_users)},
        item_id_to_idx={1000 + i: i for i in range(num_items)},
        user_sequences=user_sequences,
    )


@pytest.fixture
def tiny_data() -> PreparedData:
    return make_tiny_data()


def _seed_tiny_version(version: str = "20260101-000000") -> None:
    """Save a small but structurally complete versioned model bundle."""
    num_users, num_items = 5, 8
    gmf_dim, mlp_layers = 4, (8, 4)
    model = NeuMF(num_users, num_items, gmf_dim, mlp_layers, 0.0)

    genres_pool = ["Action", "Comedy", "Drama", "Sci-Fi"]
    movies = {
        i: {
            "movie_id": 1000 + i,
            "title": f"Test Movie {i}",
            "genres": [genres_pool[i % len(genres_pool)]],
        }
        for i in range(num_items)
    }
    metrics = {"hr@10": 0.12, "ndcg@10": 0.06}
    bundle = {
        "version": version,
        "created_at": "2026-01-01T00:00:00+00:00",
        "config": {
            "num_users": num_users,
            "num_items": num_items,
            "gmf_dim": gmf_dim,
            "mlp_layers": list(mlp_layers),
            "dropout": 0.0,
            "dataset": "ml-100k",
            "min_positive_rating": 4.0,
        },
        "state_dict": model.state_dict(),
        "user_id_to_idx": {100 + u: u for u in range(num_users)},
        "item_id_to_idx": {1000 + i: i for i in range(num_items)},
        "movies": movies,
        "user_seen": {0: [0, 1], 1: [2], 2: [3, 4, 5]},
        "metrics": metrics,
    }
    meta = {
        "version": version,
        "created_at": bundle["created_at"],
        "dataset": "ml-100k",
        "num_users": num_users,
        "num_items": num_items,
        "metrics": metrics,
    }
    registry.save_version(bundle, meta)


@pytest.fixture
def trained_app(tmp_path, monkeypatch):
    """Point the recommender at a freshly seeded tiny model version."""
    monkeypatch.setattr(settings, "artifacts_dir", str(tmp_path))
    _seed_tiny_version()
    recommender.reset_recommender()
    collector.reset()
    yield
    recommender.reset_recommender()


@pytest.fixture
def untrained_app(tmp_path, monkeypatch):
    """Point the recommender at an empty dir (no artifact present)."""
    monkeypatch.setattr(settings, "artifacts_dir", str(tmp_path))
    recommender.reset_recommender()
    collector.reset()
    yield
    recommender.reset_recommender()

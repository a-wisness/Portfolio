"""Tests for the training harness and the BPR dataset."""

import numpy as np
import pytest
import torch

from app.config import settings
from app.dataset import BPRDataset, InteractionDataset, NegativeSampler
from app.training import train_model

K = settings.top_k


def test_train_popularity_returns_metrics_no_model(tiny_data):
    model, history, metrics = train_model("popularity", tiny_data, settings)
    assert model is None
    assert len(history) == 1
    assert f"ndcg@{K}" in metrics


def test_train_mf_bce_produces_model_and_history(tiny_data):
    model, history, metrics = train_model("mf", tiny_data, settings, objective="bce", epochs=3)
    assert model is not None
    assert len(history) == 3
    assert {f"hr@{K}", f"ndcg@{K}"} <= metrics.keys()
    # Returned model is usable for scoring.
    out = model(torch.tensor([0]), torch.tensor([1]))
    assert out.shape == (1,)


def test_train_neumf_bpr_runs(tiny_data):
    model, history, metrics = train_model("neumf", tiny_data, settings, objective="bpr", epochs=2)
    assert model is not None
    assert len(history) == 2
    assert all("loss" in h for h in history)


def test_train_unknown_objective_raises(tiny_data):
    with pytest.raises(ValueError):
        train_model("mf", tiny_data, settings, objective="hinge", epochs=1)


def test_bpr_dataset_shapes_and_valid_negatives():
    train_pairs = np.array([[0, 1], [1, 2], [0, 3]], dtype=np.int64)
    user_seen = {0: {1, 3}, 1: {2}}
    ds = BPRDataset(train_pairs, user_seen, num_items=20, seed=0)
    assert len(ds) == 3
    user, pos, neg = ds[0]
    assert user.item() == 0 and pos.item() == 1
    # Negative must be an item the user hasn't seen.
    for idx in range(len(ds)):
        u, _p, n = ds[idx]
        assert n.item() not in user_seen.get(u.item(), set())


def test_bpr_dataset_resample_changes_negatives():
    train_pairs = np.array([[0, 1]], dtype=np.int64)
    ds = BPRDataset(train_pairs, {0: {1}}, num_items=500, seed=1)
    first = ds.neg_items.copy()
    ds.resample()
    assert not np.array_equal(first, ds.neg_items)


# --- Negative sampling strategies --------------------------------------- #
def test_uniform_sampler_in_range():
    s = NegativeSampler(10, np.random.default_rng(0), "uniform")
    draws = [s.draw() for _ in range(200)]
    assert all(0 <= d < 10 for d in draws)


def test_popularity_sampler_prefers_popular_items():
    counts = np.array([100, 1, 1, 1], dtype=np.int64)  # item 0 dominates
    s = NegativeSampler(4, np.random.default_rng(0), "popularity", counts, beta=1.0)
    draws = [s.draw() for _ in range(2000)]
    assert max(set(draws), key=draws.count) == 0  # most-drawn negative is item 0


def test_unknown_sampling_strategy_raises():
    with pytest.raises(ValueError):
        NegativeSampler(10, np.random.default_rng(0), "bogus")


def test_interaction_dataset_popularity_negatives_valid():
    train_pairs = np.array([[0, 1], [0, 2], [1, 3]], dtype=np.int64)
    user_seen = {0: {1, 2}, 1: {3}}
    ds = InteractionDataset(
        train_pairs, user_seen, num_items=15, num_negatives=3, seed=0,
        sampling="popularity", neg_pop_beta=0.75,
    )
    # 3 positives * (1 + 3) = 12 rows
    assert len(ds) == 12
    for row in range(len(ds)):
        if ds.labels[row] == 0.0:
            assert ds.items[row] not in user_seen[int(ds.users[row])]

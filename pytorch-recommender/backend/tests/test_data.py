"""Unit tests for the data preprocessing helpers (no download needed)."""

import numpy as np
import pandas as pd

from app.config import settings
from app.data import (
    build_index_maps,
    build_user_seen,
    leave_one_out_split,
    prepare,
    sample_eval_negatives,
)


def _ratings():
    # Two users, explicit timestamps so "latest positive" is deterministic.
    return pd.DataFrame(
        {
            "user_id": [10, 10, 10, 20, 20],
            "item_id": [1, 2, 3, 2, 4],
            "rating":  [5, 4, 2, 5, 5],
            "timestamp": [100, 200, 300, 100, 200],
        }
    )


def _movies():
    return pd.DataFrame(
        {
            "item_id": [1, 2, 3, 4],
            "title": ["A", "B", "C", "D"],
            "genres": [["Drama"], ["Comedy"], ["Action"], ["Sci-Fi"]],
        }
    )


def test_build_index_maps_contiguous():
    user2idx, item2idx = build_index_maps(_ratings())
    assert sorted(user2idx.values()) == [0, 1]
    assert sorted(item2idx.values()) == [0, 1, 2, 3]
    # Sorted by raw id
    assert user2idx[10] == 0 and user2idx[20] == 1


def test_leave_one_out_holds_out_latest_positive():
    train, test = leave_one_out_split(_ratings(), min_positive_rating=4.0)
    # User 10 positives are items 1 (t100) and 2 (t200); item 3 is rating 2 (not positive).
    # Latest positive -> item 2 held out; item 1 in train.
    u10_test = test[test["user_id"] == 10]
    assert len(u10_test) == 1
    assert u10_test.iloc[0]["item_id"] == 2
    u10_train_items = set(train[train["user_id"] == 10]["item_id"])
    assert u10_train_items == {1}


def test_leave_one_out_single_positive_has_no_test_row():
    ratings = pd.DataFrame(
        {"user_id": [30], "item_id": [7], "rating": [5], "timestamp": [10]}
    )
    train, test = leave_one_out_split(ratings, 4.0)
    assert len(test) == 0
    assert len(train) == 1


def test_sample_eval_negatives_excludes_seen_and_counts():
    user_seen = {0: {0, 1, 2}}
    rng = np.random.default_rng(0)
    negs = sample_eval_negatives(user_seen, test_users=[0], num_items=20, n=5, rng=rng)
    assert len(negs[0]) == 5
    assert all(0 <= x < 20 for x in negs[0])
    assert not (set(negs[0]) & {0, 1, 2})


def test_build_user_seen_collects_all_rated_items():
    user2idx, item2idx = build_index_maps(_ratings())
    seen = build_user_seen(_ratings(), user2idx, item2idx)
    # User 10 rated items 1,2,3 -> their indices
    assert seen[user2idx[10]] == {item2idx[1], item2idx[2], item2idx[3]}


def test_prepare_end_to_end_shapes(monkeypatch):
    # Toy corpus has only 4 items; shrink eval negatives so sampling is feasible.
    monkeypatch.setattr(settings, "num_eval_negatives", 1)
    data = prepare(_ratings(), _movies(), settings)
    assert data.num_users == 2
    assert data.num_items == 4
    assert data.train_pairs.shape[1] == 2
    # Both users have >= 2 positives -> both held out for test
    assert set(data.test_items.keys()) == {0, 1}
    # Each test user has at least one eval negative, none exceeding the request
    assert all(1 <= len(v) <= 1 for v in data.eval_negatives.values())
    # Movie metadata is keyed by item index and carries genres
    assert data.movies[0]["genres"] == ["Drama"]

"""Unit tests for the ranking metrics."""

import math

from app.metrics import hit_ratio_at_k, mean_metrics, ndcg_at_k


def test_hit_ratio_target_in_top_k():
    ranked = [5, 2, 9, 1, 7]
    assert hit_ratio_at_k(ranked, target=9, k=3) == 1.0
    assert hit_ratio_at_k(ranked, target=1, k=3) == 0.0  # rank 3 (0-based) -> outside top 3


def test_hit_ratio_target_absent():
    assert hit_ratio_at_k([1, 2, 3], target=99, k=3) == 0.0


def test_ndcg_rank_zero_is_one():
    # Target first -> 1/log2(0+2) = 1.0
    assert ndcg_at_k([7, 1, 2], target=7, k=10) == 1.0


def test_ndcg_known_value_at_rank_two():
    # Target at 0-based rank 2 -> 1/log2(4)
    expected = 1.0 / math.log2(4)
    assert math.isclose(ndcg_at_k([0, 1, 5, 9], target=5, k=10), expected)


def test_ndcg_outside_k_is_zero():
    assert ndcg_at_k([0, 1, 2, 3, 5], target=5, k=3) == 0.0


def test_mean_metrics_averages_over_users():
    rankings = {
        0: [9, 1, 2],   # target 9 at rank 0 -> hit, ndcg 1.0
        1: [1, 2, 3],   # target 9 absent -> miss, ndcg 0.0
    }
    targets = {0: 9, 1: 9}
    out = mean_metrics(rankings, targets, k=3)
    assert out["hr@3"] == 0.5
    assert math.isclose(out["ndcg@3"], 0.5)


def test_mean_metrics_empty():
    out = mean_metrics({}, {}, k=10)
    assert out == {"hr@10": 0.0, "ndcg@10": 0.0}

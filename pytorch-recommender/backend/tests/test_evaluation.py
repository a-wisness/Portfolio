"""Tests for the model-agnostic ranking evaluation and the popularity baseline."""

from app.baselines import PopularityScorer
from app.evaluation import evaluate_ranking, torch_score_fn
from app.model import NeuMF


def _data_with(test_items, eval_negatives):
    """Minimal duck-typed stand-in carrying just what evaluate_ranking reads."""
    class _D:
        pass
    d = _D()
    d.test_items = test_items
    d.eval_negatives = eval_negatives
    return d


def test_evaluate_ranking_perfect_scorer():
    data = _data_with({0: 7}, {0: [1, 2, 3]})
    # Always score the target (7) highest.
    score_fn = lambda u, cands: [10.0 if c == 7 else 0.0 for c in cands]
    out = evaluate_ranking(score_fn, data, k=10)
    assert out["hr@10"] == 1.0
    assert out["ndcg@10"] == 1.0


def test_evaluate_ranking_worst_scorer_misses_at_small_k():
    data = _data_with({0: 7}, {0: [1, 2, 3]})
    # Score the target lowest -> it ranks last among 4 candidates.
    score_fn = lambda u, cands: [0.0 if c == 7 else 5.0 for c in cands]
    out = evaluate_ranking(score_fn, data, k=2)
    assert out["hr@2"] == 0.0
    assert out["ndcg@2"] == 0.0


def test_popularity_scorer_prefers_frequent_items(tiny_data):
    scorer = PopularityScorer(tiny_data)
    fn = scorer.score_fn()
    # Item appearing in more training pairs must score higher.
    counts = scorer.popularity
    a, b = int(counts.argmax()), int(counts.argmin())
    scores = fn(user=0, candidates=[a, b])
    assert scores[0] >= scores[1]


def test_popularity_runs_through_evaluate(tiny_data):
    scorer = PopularityScorer(tiny_data)
    out = evaluate_ranking(scorer.score_fn(), tiny_data, k=10)
    assert set(out.keys()) == {"hr@10", "ndcg@10"}
    assert 0.0 <= out["hr@10"] <= 1.0


def test_torch_score_fn_returns_one_score_per_candidate(tiny_data):
    import torch

    model = NeuMF(tiny_data.num_users, tiny_data.num_items, gmf_dim=8, mlp_layers=(16, 8))
    fn = torch_score_fn(model, torch.device("cpu"))
    scores = fn(0, [1, 2, 3, 4])
    assert len(scores) == 4

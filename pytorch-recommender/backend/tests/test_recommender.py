"""Tests for the Recommender's retrieval and caching behavior.

Uses the `trained_app` fixture (a tiny serialized NeuMF artifact) and pulls the
loaded Recommender directly via get_recommender().
"""

from app.recommender import get_recommender


def test_recommender_loads_and_builds_index(trained_app):
    rec = get_recommender()
    assert rec.loaded
    assert rec.item_embeddings.shape[0] == rec.num_items
    # An ANN index was built over the item embeddings.
    assert hasattr(rec.index, "search")


def test_user_recommendations_are_cached(trained_app):
    rec = get_recommender()
    first = rec.recommend_for_user(100, top_k=3)
    assert (100, 3) in rec._user_cache
    second = rec.recommend_for_user(100, top_k=3)
    assert first == second
    assert first is rec._user_cache[(100, 3)]  # same object returned from cache


def test_similar_results_are_cached(trained_app):
    rec = get_recommender()
    rec.similar_movies(1000, top_k=3)
    assert (1000, 3) in rec._similar_cache


def test_likes_cache_key_is_order_independent(trained_app):
    rec = get_recommender()
    a = rec.recommend_for_likes([1000, 1001], top_k=3)
    b = rec.recommend_for_likes([1001, 1000], top_k=3)
    # Same set of likes -> same cache entry -> identical result object.
    assert a is b


def test_similar_excludes_query_and_respects_k(trained_app):
    rec = get_recommender()
    out = rec.similar_movies(1000, top_k=3)
    assert len(out) <= 3
    assert all(m["movie_id"] != 1000 for m in out)


def test_genre_matrix_built(trained_app):
    rec = get_recommender()
    # conftest assigns 4 distinct genres across the 8 catalog items.
    assert rec.genre_matrix is not None
    assert rec.genre_matrix.shape == (8, 4)


def test_cold_start_blend_returns_valid_results(trained_app, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "genre_weight", 0.5)  # blend active
    rec = get_recommender()
    out = rec.recommend_for_likes([1000, 1004], top_k=3)  # both "Action" in conftest
    assert 1 <= len(out) <= 3
    assert all("score" in m for m in out)
    assert 1000 not in {m["movie_id"] for m in out}

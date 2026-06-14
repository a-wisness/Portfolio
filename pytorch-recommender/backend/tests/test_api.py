"""End-to-end API tests via FastAPI's TestClient.

`trained_app` points the recommender at a tiny serialized NeuMF artifact, so the
real model-load + scoring paths run; `untrained_app` points it at an empty dir.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_reports_loaded(trained_app):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["num_items"] == 8
    assert body["version"] == "20260101-000000"
    assert "ndcg@10" in body["metrics"]


def test_health_reports_not_loaded(untrained_app):
    body = client.get("/api/health").json()
    assert body["model_loaded"] is False
    assert body["num_items"] is None


def test_endpoints_503_when_untrained(untrained_app):
    assert client.get("/api/movies").status_code == 503
    assert client.post("/api/recommend", json={"liked_movie_ids": [1000]}).status_code == 503


def test_list_and_search_movies(trained_app):
    all_movies = client.get("/api/movies").json()
    assert len(all_movies) == 8
    assert {"movie_id", "title", "genres"} <= all_movies[0].keys()

    filtered = client.get("/api/movies", params={"search": "Movie 3"}).json()
    assert len(filtered) == 1
    assert filtered[0]["movie_id"] == 1003


def test_recommend_from_likes(trained_app):
    res = client.post("/api/recommend", json={"liked_movie_ids": [1000, 1001], "top_k": 3})
    assert res.status_code == 200
    body = res.json()
    assert body["strategy"].startswith("item-embedding")
    recs = body["recommendations"]
    assert 1 <= len(recs) <= 3
    returned_ids = {r["movie_id"] for r in recs}
    # Liked movies must be excluded from their own recommendations
    assert 1000 not in returned_ids and 1001 not in returned_ids
    assert all("score" in r for r in recs)


def test_recommend_unknown_movie_ids_404(trained_app):
    res = client.post("/api/recommend", json={"liked_movie_ids": [99999]})
    assert res.status_code == 404


def test_recommend_requires_at_least_one_id(trained_app):
    res = client.post("/api/recommend", json={"liked_movie_ids": []})
    assert res.status_code == 422  # Pydantic min_length


def test_user_recommendations(trained_app):
    res = client.get("/api/users/100/recommendations", params={"top_k": 3})
    assert res.status_code == 200
    body = res.json()
    assert body["strategy"] == "neural collaborative filtering"
    # User idx 0 has seen items [0, 1] -> movie_ids 1000, 1001 must be excluded
    ids = {r["movie_id"] for r in body["recommendations"]}
    assert 1000 not in ids and 1001 not in ids


def test_user_recommendations_unknown_user_404(trained_app):
    assert client.get("/api/users/9999/recommendations").status_code == 404


def test_similar_movies(trained_app):
    res = client.get("/api/movies/1000/similar", params={"top_k": 3})
    assert res.status_code == 200
    recs = res.json()["recommendations"]
    assert 1 <= len(recs) <= 3
    assert 1000 not in {r["movie_id"] for r in recs}  # excludes itself


def test_similar_unknown_movie_404(trained_app):
    assert client.get("/api/movies/99999/similar").status_code == 404


# --- Operational endpoints (Phase 4) ------------------------------------- #
def test_list_models(trained_app):
    models = client.get("/api/models").json()
    assert len(models) == 1
    assert models[0]["version"] == "20260101-000000"
    assert models[0]["active"] is True


def test_activate_known_version(trained_app):
    res = client.post("/api/models/20260101-000000/activate")
    assert res.status_code == 200
    body = res.json()
    assert body["active_version"] == "20260101-000000"
    assert body["model_loaded"] is True


def test_activate_unknown_version_404(trained_app):
    assert client.post("/api/models/nope/activate").status_code == 404


def test_reload_model(trained_app):
    res = client.post("/api/models/reload")
    assert res.status_code == 200
    assert res.json()["model_loaded"] is True


def test_metrics_endpoint_tracks_requests_and_coverage(trained_app):
    # Generate a recommendation so coverage is non-zero.
    client.post("/api/recommend", json={"liked_movie_ids": [1000, 1001], "top_k": 3})
    snap = client.get("/api/metrics").json()
    assert snap["total_requests"] >= 1
    assert snap["recommendations"]["catalog_size"] == 8
    assert snap["recommendations"]["distinct_items_recommended"] >= 1
    assert snap["recommendations"]["coverage"] > 0.0
    assert snap["latency_ms"]["count"] >= 1

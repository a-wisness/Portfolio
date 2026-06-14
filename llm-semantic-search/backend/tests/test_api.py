"""End-to-end API tests via FastAPI's TestClient.

Exercises the full HTTP surface and the ingest -> index -> search -> answer
pipeline with the test doubles wired up in conftest.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_search_without_documents_returns_400():
    res = client.post("/api/search", json={"query": "anything"})
    assert res.status_code == 400


def test_ingest_indexes_chunks():
    files = {"file": ("policy.txt", b"Refunds are allowed within 30 days of purchase.", "text/plain")}
    res = client.post("/api/ingest", files=files)
    assert res.status_code == 200
    body = res.json()
    assert body["filename"] == "policy.txt"
    assert body["chunks_indexed"] >= 1


def test_ingest_rejects_empty_file():
    files = {"file": ("empty.txt", b"", "text/plain")}
    res = client.post("/api/ingest", files=files)
    assert res.status_code == 400


def test_ingest_rejects_unsupported_type():
    files = {"file": ("data.bin", b"\x00\x01\x02", "application/octet-stream")}
    res = client.post("/api/ingest", files=files)
    assert res.status_code == 400


def test_full_search_flow():
    content = b"The mitochondria is the powerhouse of the cell. It produces ATP."
    client.post("/api/ingest", files={"file": ("bio.txt", content, "text/plain")})

    res = client.post("/api/search", json={"query": "what makes ATP in the cell"})
    assert res.status_code == 200
    body = res.json()
    assert body["query"] == "what makes ATP in the cell"
    assert isinstance(body["answer"], str) and body["answer"]
    assert len(body["sources"]) >= 1
    src = body["sources"][0]
    assert {"index", "filename", "chunk_index", "text", "score"} <= src.keys()
    assert src["index"] == 1


def test_search_validates_empty_query():
    client.post("/api/ingest", files={"file": ("d.txt", b"some content here", "text/plain")})
    res = client.post("/api/search", json={"query": ""})
    assert res.status_code == 422  # Pydantic min_length violation


def test_stats_reflects_ingested_documents():
    client.post("/api/ingest", files={"file": ("a.txt", b"alpha content", "text/plain")})
    client.post("/api/ingest", files={"file": ("b.txt", b"beta content", "text/plain")})
    res = client.get("/api/stats")
    assert res.status_code == 200
    body = res.json()
    names = {d["filename"] for d in body["documents"]}
    assert {"a.txt", "b.txt"} <= names
    assert body["total_chunks"] >= 2


def test_reset_clears_index():
    client.post("/api/ingest", files={"file": ("x.txt", b"to be cleared", "text/plain")})
    assert client.post("/api/reset").json() == {"status": "cleared"}
    assert client.get("/api/stats").json()["total_chunks"] == 0

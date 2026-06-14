"""Tests for the model registry (versioning + active pointer)."""

import pytest

from app import registry
from app.config import settings


@pytest.fixture
def tmp_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "artifacts_dir", str(tmp_path))
    return tmp_path


def _save(version: str):
    # registry.save_version torch-saves the bundle as-is; contents are opaque to it.
    registry.save_version({"dummy": True}, {"version": version, "created_at": version,
                                            "metrics": {"ndcg@10": 0.1}})


def test_new_version_id_is_unique(tmp_artifacts):
    v1 = registry.new_version_id()
    _save(v1)
    v2 = registry.new_version_id()
    assert v1 != v2  # second id disambiguated even within the same second


def test_save_sets_active_and_resolves(tmp_artifacts):
    _save("20260101-000000")
    assert registry.get_active_version() == "20260101-000000"
    path = registry.resolve_active_path()
    assert path is not None and path.exists()
    assert (tmp_artifacts / "versions" / "20260101-000000.json").exists()


def test_list_versions_flags_active_and_sorts(tmp_artifacts):
    _save("20260101-000000")
    _save("20260102-000000")  # newer; becomes active
    versions = registry.list_versions()
    assert [v["version"] for v in versions] == ["20260102-000000", "20260101-000000"]
    assert versions[0]["active"] is True
    assert versions[1]["active"] is False


def test_set_active_switches(tmp_artifacts):
    _save("20260101-000000")
    _save("20260102-000000")
    registry.set_active("20260101-000000")
    assert registry.get_active_version() == "20260101-000000"


def test_set_active_unknown_raises(tmp_artifacts):
    with pytest.raises(FileNotFoundError):
        registry.set_active("does-not-exist")


def test_legacy_fallback(tmp_artifacts):
    # No versions/ dir, but a legacy single model.pt present.
    (tmp_artifacts / settings.artifact_name).write_bytes(b"not-a-real-model")
    assert registry.get_active_version() is None
    assert registry.resolve_active_path() == tmp_artifacts / settings.artifact_name
    assert registry.list_versions() == [{"version": "legacy", "active": True}]

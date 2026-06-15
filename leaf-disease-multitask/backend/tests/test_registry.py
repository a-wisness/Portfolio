"""Registry tests — no TensorFlow (pure filesystem + JSON)."""
from __future__ import annotations

import json

from app.config import Settings
from app.registry import Registry


def _settings(tmp_path) -> Settings:
    return Settings(artifacts_dir=tmp_path / "artifacts")


def _fake_model(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-keras-model")


def test_create_finalize_and_activate(tmp_path):
    reg = Registry(_settings(tmp_path))
    version, vdir = reg.create_version("v1")
    _fake_model(vdir / "model.keras")
    reg.finalize("v1", ["a", "b"], {"created_utc": "t", "num_classes": 2,
                                    "val_metrics": {"loss": 1.0}})

    assert reg.active_version() == "v1"
    files = reg.resolve_active()
    assert files["version"] == "v1"
    assert json.loads(files["labels"].read_text()) == ["a", "b"]

    listing = reg.list_versions()
    assert len(listing) == 1
    assert listing[0]["version"] == "v1" and listing[0]["active"] is True
    assert listing[0]["num_classes"] == 2


def test_multiple_versions_and_switch(tmp_path):
    reg = Registry(_settings(tmp_path))
    for v in ("v1", "v2"):
        _, vdir = reg.create_version(v)
        _fake_model(vdir / "model.keras")
        reg.finalize(v, ["a"], {"num_classes": 1})

    # Latest finalized (v2) is active.
    assert reg.active_version() == "v2"
    reg.set_active("v1")
    assert reg.active_version() == "v1"
    assert sum(1 for x in reg.list_versions() if x["active"]) == 1


def test_set_active_unknown_raises(tmp_path):
    reg = Registry(_settings(tmp_path))
    import pytest

    with pytest.raises(FileNotFoundError):
        reg.set_active("nope")


def test_has_any_false_when_empty(tmp_path):
    reg = Registry(_settings(tmp_path))
    assert reg.has_any() is False
    assert reg.resolve_active() is None


def test_legacy_migration(tmp_path):
    settings = _settings(tmp_path)
    # Simulate a Phase 1/2 flat artifact.
    _fake_model(settings.model_path)
    settings.labels_path.write_text(json.dumps(["a", "b", "c"]))
    settings.metadata_path.write_text(json.dumps({"num_classes": 3}))

    reg = Registry(settings)
    reg.ensure_initialized()

    assert reg.active_version() == "legacy"
    files = reg.resolve_active()
    assert files["model"].exists()
    assert json.loads(files["labels"].read_text()) == ["a", "b", "c"]
    # Flat files were moved, not left behind.
    assert not settings.model_path.exists()

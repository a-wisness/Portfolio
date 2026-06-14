"""Model registry: versioned artifacts + the active-version pointer.

Each training run writes a snapshot to ``artifacts/versions/<version>.pt`` with a
small JSON sidecar (``<version>.json``) holding its metadata (created-at,
dataset, metrics). ``artifacts/active.txt`` names the version the API serves.

This lets the service list versions, switch the active model, and reload — all
without a redeploy. A legacy single ``artifacts/model.pt`` (pre-versioning) is
still honored as a fallback when no versions exist.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from .config import settings


def artifacts_dir() -> Path:
    return Path(settings.artifacts_dir)


def versions_dir() -> Path:
    return artifacts_dir() / settings.versions_subdir


def active_file() -> Path:
    return artifacts_dir() / settings.active_pointer


def legacy_path() -> Path:
    return artifacts_dir() / settings.artifact_name


def _version_pt(version: str) -> Path:
    return versions_dir() / f"{version}.pt"


def _version_json(version: str) -> Path:
    return versions_dir() / f"{version}.json"


def new_version_id() -> str:
    """A sortable UTC-timestamp id, disambiguated if two land in one second."""
    base = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    version, n = base, 1
    while _version_pt(version).exists():
        n += 1
        version = f"{base}-{n}"
    return version


def save_version(bundle: dict, meta: dict) -> str:
    """Persist a model bundle + metadata sidecar and make it the active version."""
    versions_dir().mkdir(parents=True, exist_ok=True)
    version = meta["version"]
    torch.save(bundle, _version_pt(version))
    _version_json(version).write_text(json.dumps(meta, indent=2))
    set_active(version)
    return version


def set_active(version: str) -> None:
    if not _version_pt(version).exists():
        raise FileNotFoundError(f"Unknown model version: {version!r}")
    active_file().write_text(version.strip() + "\n")


def get_active_version() -> str | None:
    if active_file().exists():
        version = active_file().read_text().strip()
        if version and _version_pt(version).exists():
            return version
    return None


def resolve_active_path() -> Path | None:
    """Filesystem path the recommender should load, or None if nothing trained."""
    version = get_active_version()
    if version:
        return _version_pt(version)
    if legacy_path().exists():  # pre-versioning fallback
        return legacy_path()
    return None


def list_versions() -> list[dict]:
    """Metadata for every known version, newest first, with the active one flagged."""
    active = get_active_version()
    entries: list[dict] = []
    if versions_dir().exists():
        for sidecar in versions_dir().glob("*.json"):
            try:
                meta = json.loads(sidecar.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            meta["active"] = meta.get("version") == active
            entries.append(meta)
    if not entries and legacy_path().exists():
        entries.append({"version": "legacy", "active": True})
    entries.sort(key=lambda m: m.get("created_at", ""), reverse=True)
    return entries

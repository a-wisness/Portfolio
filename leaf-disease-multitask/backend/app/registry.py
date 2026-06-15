"""A tiny on-disk model registry: versioned artifacts + an ``active`` pointer.

Layout::

    artifacts/
      active.txt                 # names the active version
      versions/
        20260615-165356/
          model.keras
          labels.json
          metadata.json

Kept free of TensorFlow — it only manages directories, the pointer file, and the
JSON sidecars. The caller saves the Keras model file into the version directory.
Legacy flat artifacts (``artifacts/leaflens_model.keras`` from Phase 1–2) are
migrated into ``versions/legacy/`` on first use, so older runs keep serving.
"""
from __future__ import annotations

import datetime as _dt
import json
import shutil
from pathlib import Path

from .config import Settings, get_settings

MODEL_NAME = "model.keras"
LABELS_NAME = "labels.json"
METADATA_NAME = "metadata.json"


class Registry:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    # --- paths ------------------------------------------------------------
    @property
    def versions_dir(self) -> Path:
        return self.settings.versions_dir

    @property
    def pointer_path(self) -> Path:
        return self.settings.active_pointer_path

    def version_dir(self, version: str) -> Path:
        return self.versions_dir / version

    def version_files(self, version: str) -> dict[str, Path]:
        d = self.version_dir(version)
        return {
            "model": d / MODEL_NAME,
            "labels": d / LABELS_NAME,
            "metadata": d / METADATA_NAME,
        }

    # --- listing / active -------------------------------------------------
    def list_versions(self) -> list[dict]:
        self.ensure_initialized()
        out: list[dict] = []
        if not self.versions_dir.is_dir():
            return out
        active = self.active_version()
        for d in sorted(self.versions_dir.iterdir()):
            if not d.is_dir() or not (d / MODEL_NAME).exists():
                continue
            meta = {}
            meta_path = d / METADATA_NAME
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                except (ValueError, OSError):
                    meta = {}
            out.append({
                "version": d.name,
                "active": d.name == active,
                "created_utc": meta.get("created_utc"),
                "val_metrics": meta.get("val_metrics"),
                "num_classes": meta.get("num_classes"),
            })
        return out

    def active_version(self) -> str | None:
        if self.pointer_path.exists():
            name = self.pointer_path.read_text().strip()
            if name and (self.version_dir(name) / MODEL_NAME).exists():
                return name
        # Fall back to the most recent valid version.
        versions = [
            d.name for d in sorted(self.versions_dir.iterdir())
            if d.is_dir() and (d / MODEL_NAME).exists()
        ] if self.versions_dir.is_dir() else []
        return versions[-1] if versions else None

    def set_active(self, version: str) -> None:
        if not (self.version_dir(version) / MODEL_NAME).exists():
            raise FileNotFoundError(f"No such model version: {version}")
        self.pointer_path.parent.mkdir(parents=True, exist_ok=True)
        self.pointer_path.write_text(version)

    # --- writing ----------------------------------------------------------
    def create_version(self, version: str | None = None) -> tuple[str, Path]:
        """Make a fresh version directory and return ``(version_id, dir)``."""
        version = version or _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
        vdir = self.version_dir(version)
        vdir.mkdir(parents=True, exist_ok=True)
        return version, vdir

    def finalize(
        self, version: str, labels: list[str], metadata: dict,
        set_active: bool = True,
    ) -> None:
        """Write the label map + metadata sidecar; optionally activate."""
        files = self.version_files(version)
        if not files["model"].exists():
            raise FileNotFoundError(
                f"Model file missing for version {version}; save it before finalize()."
            )
        files["labels"].write_text(json.dumps(labels, indent=2))
        files["metadata"].write_text(json.dumps(metadata, indent=2))
        if set_active:
            self.set_active(version)

    # --- resolution / migration ------------------------------------------
    def has_any(self) -> bool:
        self.ensure_initialized()
        return self.active_version() is not None

    def resolve_active(self) -> dict[str, Path] | None:
        """Files for the active version, or None if nothing is registered."""
        self.ensure_initialized()
        version = self.active_version()
        if version is None:
            return None
        files = self.version_files(version)
        files["version"] = version  # type: ignore[assignment]
        return files

    def ensure_initialized(self) -> None:
        """Migrate legacy flat artifacts into ``versions/legacy/`` once."""
        has_versions = self.versions_dir.is_dir() and any(
            d.is_dir() and (d / MODEL_NAME).exists()
            for d in self.versions_dir.iterdir()
        )
        legacy_model = self.settings.model_path
        if has_versions or not legacy_model.exists():
            return
        _, vdir = self.create_version("legacy")
        shutil.move(str(legacy_model), str(vdir / MODEL_NAME))
        for src, dst in (
            (self.settings.labels_path, vdir / LABELS_NAME),
            (self.settings.metadata_path, vdir / METADATA_NAME),
        ):
            if src.exists():
                shutil.move(str(src), str(dst))
        self.set_active("legacy")

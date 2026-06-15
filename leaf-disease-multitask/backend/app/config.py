"""Typed configuration for LeafLens.

All paths and hyperparameters are config-driven and overridable via environment
variables (prefix ``LEAFLENS_``) or a ``.env`` file. Kept free of TensorFlow so
it can be imported by the API and tests without pulling in heavy deps.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo layout anchors -------------------------------------------------------
# backend/app/config.py -> backend/app -> backend -> project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
# The original notebook project lives next to this one and holds the data.
_DEFAULT_DATA_ROOT = _PROJECT_ROOT.parent / "LeafDiseaseDetection-DL"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LEAFLENS_",
        env_file=".env",
        extra="ignore",
    )

    # --- Data locations (referenced in place; not copied) ------------------
    classification_dir: Path = Field(
        default=_DEFAULT_DATA_ROOT / "Classification Data",
        description="Folder with train/ and test/ subfolders, one dir per class.",
    )
    segmentation_dir: Path = Field(
        default=_DEFAULT_DATA_ROOT / "Image Segmentation Data",
        description="Folder with data/{images,masks} and aug_data/{images,masks}.",
    )

    # --- Image / model geometry -------------------------------------------
    img_height: int = 224
    img_width: int = 224
    img_channels: int = 3

    # --- Training hyperparameters -----------------------------------------
    seed: int = 42
    batch_size: int = 16
    val_split: float = 0.2
    # Two-phase transfer learning.
    head_epochs: int = 12          # phase A: encoder frozen
    finetune_epochs: int = 8       # phase B: top encoder blocks unfrozen
    head_lr: float = 1e-3
    finetune_lr: float = 1e-5
    finetune_at: int = 100         # unfreeze encoder layers from this index up
    early_stopping_patience: int = 6
    # Optional cap on batches/epoch. With the shuffled, repeated train stream
    # this trades full-pass epochs for faster ones (useful for bounded runs);
    # None means one full pass over the data.
    max_steps_per_epoch: int | None = None

    # Multi-task mixing + loss weighting.
    seg_sample_ratio: float = 0.5  # fraction of each batch drawn from seg data
    seg_loss_weight: float = 1.0
    cls_loss_weight: float = 1.0
    dropout: float = 0.2

    # Augmentation + class imbalance (Phase 2).
    augment: bool = True           # task-correct augmentation on the train split
    use_class_weights: bool = True  # balance the 28 classes (one has only 2 imgs)
    max_class_weight: float = 10.0  # cap so a tiny class can't dominate the loss

    # --- Artifacts ---------------------------------------------------------
    artifacts_dir: Path = Field(default=_PROJECT_ROOT / "artifacts")
    model_filename: str = "leaflens_model.keras"
    labels_filename: str = "labels.json"
    metadata_filename: str = "metadata.json"

    # --- Inference ---------------------------------------------------------
    mask_threshold: float = 0.5    # sigmoid prob above this = leaf
    top_k: int = 3
    cache_size: int = 64           # bounded LRU of recent predictions (0 = off)

    @property
    def versions_dir(self) -> Path:
        return self.artifacts_dir / "versions"

    @property
    def active_pointer_path(self) -> Path:
        return self.artifacts_dir / "active.txt"

    @property
    def image_size(self) -> tuple[int, int]:
        return (self.img_height, self.img_width)

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (self.img_height, self.img_width, self.img_channels)

    @property
    def model_path(self) -> Path:
        return self.artifacts_dir / self.model_filename

    @property
    def labels_path(self) -> Path:
        return self.artifacts_dir / self.labels_filename

    @property
    def metadata_path(self) -> Path:
        return self.artifacts_dir / self.metadata_filename


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()

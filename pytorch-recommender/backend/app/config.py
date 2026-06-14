"""Typed settings and hyperparameters, loaded from environment / .env.

Every knob the training run and the API need lives here so runs are
reproducible and configurable without code edits.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Paths
    data_dir: str = "./data"
    artifacts_dir: str = "./artifacts"
    artifact_name: str = "model.pt"        # legacy single-file fallback
    versions_subdir: str = "versions"      # versioned artifacts live here
    active_pointer: str = "active.txt"     # names the currently-active version

    # Dataset
    dataset: str = "ml-100k"            # "ml-100k" or "ml-1m"
    min_positive_rating: float = 4.0    # rating >= this counts as a "like"

    # Model (NeuMF)
    gmf_dim: int = 32
    mlp_layers: tuple[int, ...] = (64, 32, 16, 8)  # layers[0] = 2 * mlp_embed_dim
    dropout: float = 0.0

    # Training
    num_negatives: int = 4              # negatives per positive (training)
    num_eval_negatives: int = 99        # candidates besides the held-out item
    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 1e-3
    seed: int = 42
    negative_sampling: str = "uniform"  # "uniform" or "popularity" (hard negatives)
    neg_pop_beta: float = 0.75          # popularity exponent when sampling negatives

    # SASRec (sequential model — experiment)
    sas_max_len: int = 50
    sas_dim: int = 64
    sas_heads: int = 2
    sas_blocks: int = 2
    sas_dropout: float = 0.2

    # Cold-start blend
    genre_weight: float = 0.2           # weight on genre-prior vs. embedding similarity

    # Serving / evaluation
    top_k: int = 10
    use_faiss: bool = False   # use a FAISS ANN index when available (else brute force)
    cache_size: int = 512     # max entries per recommendation LRU cache

    @property
    def mlp_embed_dim(self) -> int:
        return self.mlp_layers[0] // 2


settings = Settings()

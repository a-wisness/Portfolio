"""Inference-time recommender: loads the active model version and ranks movies.

Strategies served by the API:
  * recommend_for_likes  — cold-start for an ad-hoc visitor: aggregate the
    learned item embeddings of the liked movies and retrieve nearest items.
  * recommend_for_user   — true NeuMF scoring for a known training user.
  * similar_movies       — nearest items in the learned embedding space.

The active artifact is resolved through the model registry, and the loaded
recommender is held in a module-level slot that can be hot-swapped
(`reload_recommender`) without a process restart. Per-query results are memoized
in bounded LRU caches (recommendations are deterministic for a loaded model).
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import registry
from .ann import build_index
from .config import settings
from .model import NeuMF


class ModelNotLoaded(Exception):
    """Raised when an operation needs a trained artifact that isn't present."""


class UnknownUser(Exception):
    pass


class LRUCache:
    """Tiny bounded LRU cache (insertion/access-ordered OrderedDict)."""

    def __init__(self, maxsize: int) -> None:
        self._data: OrderedDict[Any, Any] = OrderedDict()
        self._max = max(1, maxsize)

    def get(self, key: Any) -> Any | None:
        if key in self._data:
            self._data.move_to_end(key)
            return self._data[key]
        return None

    def put(self, key: Any, value: Any) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self._max:
            self._data.popitem(last=False)

    def __contains__(self, key: Any) -> bool:
        return key in self._data

    def __getitem__(self, key: Any) -> Any:
        return self._data[key]

    def __len__(self) -> int:
        return len(self._data)


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class Recommender:
    def __init__(self, artifact_path: Path | None):
        self.loaded = False
        self.version: str | None = None
        self.created_at: str | None = None
        self.num_users = 0
        self.num_items = 0
        self.metrics: dict[str, float] = {}
        if artifact_path is None or not Path(artifact_path).exists():
            return

        bundle = torch.load(artifact_path, map_location="cpu", weights_only=False)
        cfg = bundle["config"]
        self.version = bundle.get("version")
        self.created_at = bundle.get("created_at")
        self.num_users = cfg["num_users"]
        self.num_items = cfg["num_items"]
        self.metrics = bundle.get("metrics", {})

        self.model = NeuMF(
            cfg["num_users"], cfg["num_items"],
            cfg["gmf_dim"], tuple(cfg["mlp_layers"]), cfg["dropout"],
        )
        self.model.load_state_dict(bundle["state_dict"])
        self.model.eval()

        self.user_id_to_idx: dict[int, int] = bundle["user_id_to_idx"]
        self.item_id_to_idx: dict[int, int] = bundle["item_id_to_idx"]
        self.movies: dict[int, dict] = bundle["movies"]
        # JSON round-trip serializes int dict keys as strings; cast back.
        self.user_seen: dict[int, set[int]] = {
            int(u): set(items) for u, items in bundle["user_seen"].items()
        }

        emb = self.model.item_embedding_matrix().cpu().numpy()
        self.item_embeddings = _l2_normalize(emb)
        self.index = build_index(self.item_embeddings, settings.use_faiss)

        # Genre multi-hot matrix (normalized) for the cold-start genre blend.
        genres = sorted({g for m in self.movies.values() for g in m["genres"]})
        self._genre_index = {g: i for i, g in enumerate(genres)}
        if genres:
            gmat = np.zeros((self.num_items, len(genres)), dtype=np.float32)
            for idx, meta in self.movies.items():
                for g in meta["genres"]:
                    gmat[idx, self._genre_index[g]] = 1.0
            self.genre_matrix: np.ndarray | None = _l2_normalize(gmat)
        else:
            self.genre_matrix = None

        self._user_cache = LRUCache(settings.cache_size)
        self._similar_cache = LRUCache(settings.cache_size)
        self._likes_cache = LRUCache(settings.cache_size)
        self.loaded = True

    # ----------------------------------------------------------------- #
    def _require_loaded(self) -> None:
        if not self.loaded:
            raise ModelNotLoaded("No trained model artifact is available.")

    def _movie(self, item_idx: int, score: float) -> dict:
        meta = self.movies[item_idx]
        return {
            "movie_id": meta["movie_id"],
            "title": meta["title"],
            "genres": meta["genres"],
            "score": round(float(score), 4),
        }

    # ----------------------------------------------------------------- #
    def search_movies(self, query: str, limit: int) -> list[dict]:
        self._require_loaded()
        q = query.strip().lower()
        items = []
        for meta in self.movies.values():
            if not q or q in meta["title"].lower():
                items.append({
                    "movie_id": meta["movie_id"],
                    "title": meta["title"],
                    "genres": meta["genres"],
                })
        items.sort(key=lambda m: m["title"].lower())
        return items[:limit]

    def recommend_for_likes(self, liked_movie_ids: list[int], top_k: int) -> list[dict]:
        """Cold-start: retrieve items nearest to the mean of the liked embeddings."""
        self._require_loaded()
        liked_idx = [
            self.item_id_to_idx[mid]
            for mid in liked_movie_ids
            if mid in self.item_id_to_idx
        ]
        if not liked_idx:
            return []

        cache_key = (tuple(sorted(liked_idx)), top_k)
        cached = self._likes_cache.get(cache_key)
        if cached is not None:
            return cached

        profile = self.item_embeddings[liked_idx].mean(axis=0)
        norm = np.linalg.norm(profile)
        if norm > 0:
            profile = profile / norm

        liked_set = set(liked_idx)
        weight = settings.genre_weight
        blend = weight > 0 and self.genre_matrix is not None

        # Pull a wider candidate set when blending so genre re-ranking can reorder.
        fetch = top_k + len(liked_idx)
        if blend:
            fetch = max(fetch, top_k * 5 + len(liked_idx))
        ids, scores = self.index.search(profile, fetch)
        pairs = [
            (i, s) for i, s in zip(ids, scores)
            if i not in liked_set and i in self.movies
        ]

        if blend:
            gprofile = self.genre_matrix[liked_idx].mean(axis=0)
            gnorm = np.linalg.norm(gprofile)
            if gnorm > 0:  # only blend when the liked set has genre signal
                gprofile = gprofile / gnorm
                pairs = [
                    (i, (1 - weight) * s + weight * float(self.genre_matrix[i] @ gprofile))
                    for i, s in pairs
                ]
                pairs.sort(key=lambda p: p[1], reverse=True)

        out = [self._movie(i, s) for i, s in pairs[:top_k]]
        self._likes_cache.put(cache_key, out)
        return out

    @torch.no_grad()
    def recommend_for_user(self, user_id: int, top_k: int) -> list[dict]:
        """True NeuMF scoring for a known training user; excludes seen items."""
        self._require_loaded()
        if user_id not in self.user_id_to_idx:
            raise UnknownUser(f"User {user_id} is not in the training set.")

        cache_key = (user_id, top_k)
        cached = self._user_cache.get(cache_key)
        if cached is not None:
            return cached

        uidx = self.user_id_to_idx[user_id]
        items = torch.arange(self.num_items, dtype=torch.long)
        users = torch.full((self.num_items,), uidx, dtype=torch.long)
        scores = torch.sigmoid(self.model(users, items))
        for seen in self.user_seen.get(uidx, set()):
            scores[seen] = float("-inf")

        top = torch.topk(scores, k=min(top_k, self.num_items)).indices.tolist()
        out = [self._movie(i, scores[i].item()) for i in top if i in self.movies]
        self._user_cache.put(cache_key, out)
        return out

    def similar_movies(self, movie_id: int, top_k: int) -> list[dict]:
        """Nearest movies to a given movie in the learned embedding space."""
        self._require_loaded()
        if movie_id not in self.item_id_to_idx:
            return []

        cache_key = (movie_id, top_k)
        cached = self._similar_cache.get(cache_key)
        if cached is not None:
            return cached

        idx = self.item_id_to_idx[movie_id]
        ids, scores = self.index.search(self.item_embeddings[idx], top_k + 1)
        out = [
            self._movie(i, s)
            for i, s in zip(ids, scores)
            if i != idx and i in self.movies
        ][:top_k]
        self._similar_cache.put(cache_key, out)
        return out


# --------------------------------------------------------------------------- #
# Module-level active recommender — hot-swappable without a restart.
# --------------------------------------------------------------------------- #
_active: Recommender | None = None


def get_recommender() -> Recommender:
    """Return the active recommender, loading it lazily on first use."""
    global _active
    if _active is None:
        _active = Recommender(registry.resolve_active_path())
    return _active


def reload_recommender() -> Recommender:
    """Re-resolve and reload the active version (after training or activation)."""
    global _active
    _active = Recommender(registry.resolve_active_path())
    return _active


def reset_recommender() -> None:
    """Drop the cached recommender so the next get reloads (used in tests)."""
    global _active
    _active = None

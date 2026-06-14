"""MovieLens download and preprocessing.

Turns raw MovieLens ratings into the structures the training loop and evaluation
need:
  * contiguous user/item index maps,
  * a leave-one-out split (each user's most recent *positive* interaction is the
    held-out test item),
  * per-user "seen" sets (for negative sampling and excluding already-watched
    items at recommendation time),
  * a fixed set of evaluation negatives per user (the ranking candidates).

The pure helpers (index maps, split, negative sampling) take DataFrames/dicts so
they can be unit-tested without downloading anything.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_URLS = {
    "ml-100k": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
    "ml-1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
}

GENRE_COLUMNS = [
    "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def download_movielens(data_dir: str, dataset: str = "ml-100k") -> Path:
    """Download and extract a MovieLens dataset; return its extracted dir.

    Idempotent: if the dataset folder already exists, the download is skipped.
    """
    if dataset not in DATASET_URLS:
        raise ValueError(f"Unknown dataset {dataset!r}. Choose from {list(DATASET_URLS)}.")

    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    extracted = root / dataset
    if extracted.exists():
        return extracted

    import requests  # imported here so tests that stub data don't need network

    resp = requests.get(DATASET_URLS[dataset], timeout=120)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(root)
    if not extracted.exists():
        raise RuntimeError(f"Expected {extracted} after extraction.")
    return extracted


def load_ratings(dataset_dir: Path, dataset: str = "ml-100k") -> pd.DataFrame:
    """Load ratings as columns [user_id, item_id, rating, timestamp]."""
    if dataset == "ml-100k":
        path = dataset_dir / "u.data"
        df = pd.read_csv(
            path, sep="\t", names=["user_id", "item_id", "rating", "timestamp"],
            engine="python",
        )
    elif dataset == "ml-1m":
        path = dataset_dir / "ratings.dat"
        df = pd.read_csv(
            path, sep="::", names=["user_id", "item_id", "rating", "timestamp"],
            engine="python",
        )
    else:
        raise ValueError(f"Unknown dataset {dataset!r}.")
    return df


def load_movies(dataset_dir: Path, dataset: str = "ml-100k") -> pd.DataFrame:
    """Load movie metadata as columns [item_id, title, genres(list[str])]."""
    if dataset == "ml-100k":
        path = dataset_dir / "u.item"
        cols = ["item_id", "title", "release", "video_release", "imdb_url", *GENRE_COLUMNS]
        raw = pd.read_csv(path, sep="|", names=cols, encoding="latin-1", engine="python")
        genres = [
            [g for g, flag in zip(GENRE_COLUMNS, row) if flag == 1]
            for row in raw[GENRE_COLUMNS].itertuples(index=False)
        ]
        return pd.DataFrame(
            {"item_id": raw["item_id"], "title": raw["title"], "genres": genres}
        )
    elif dataset == "ml-1m":
        path = dataset_dir / "movies.dat"
        raw = pd.read_csv(
            path, sep="::", names=["item_id", "title", "genres"],
            encoding="latin-1", engine="python",
        )
        raw["genres"] = raw["genres"].apply(lambda s: s.split("|"))
        return raw
    else:
        raise ValueError(f"Unknown dataset {dataset!r}.")


# --------------------------------------------------------------------------- #
# Pure preprocessing helpers (unit-tested without any download)
# --------------------------------------------------------------------------- #
def build_index_maps(
    ratings: pd.DataFrame,
) -> tuple[dict[int, int], dict[int, int]]:
    """Map raw user/item ids to contiguous 0..N-1 indices (sorted by id)."""
    user2idx = {uid: i for i, uid in enumerate(sorted(ratings["user_id"].unique()))}
    item2idx = {iid: i for i, iid in enumerate(sorted(ratings["item_id"].unique()))}
    return user2idx, item2idx


def leave_one_out_split(
    ratings: pd.DataFrame, min_positive_rating: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split positives into (train, test).

    Positives are interactions with rating >= ``min_positive_rating``. For each
    user with >= 2 positives, the most recent positive (by timestamp) becomes
    the single held-out test row; the rest are training rows. Users with fewer
    than 2 positives contribute only training rows (no test item).
    """
    positives = ratings[ratings["rating"] >= min_positive_rating].copy()
    positives.sort_values(["user_id", "timestamp"], inplace=True)

    test_rows, train_rows = [], []
    for _, group in positives.groupby("user_id", sort=False):
        if len(group) >= 2:
            test_rows.append(group.iloc[-1])
            train_rows.append(group.iloc[:-1])
        else:
            train_rows.append(group)

    train_df = pd.concat(train_rows, ignore_index=True) if train_rows else positives.iloc[:0]
    test_df = (
        pd.DataFrame(test_rows).reset_index(drop=True)
        if test_rows
        else positives.iloc[:0]
    )
    return train_df, test_df


def build_user_seen(
    ratings: pd.DataFrame, user2idx: dict[int, int], item2idx: dict[int, int]
) -> dict[int, set[int]]:
    """For each user index, the set of item indices they have *any* rating for.

    Used both to avoid sampling negatives the user has seen and to exclude
    already-watched movies from recommendations.
    """
    seen: dict[int, set[int]] = {}
    for uid, iid in zip(ratings["user_id"], ratings["item_id"]):
        if uid in user2idx and iid in item2idx:
            seen.setdefault(user2idx[uid], set()).add(item2idx[iid])
    return seen


def sample_eval_negatives(
    user_seen: dict[int, set[int]],
    test_users: list[int],
    num_items: int,
    n: int,
    rng: np.random.Generator,
) -> dict[int, list[int]]:
    """Sample ``n`` unseen item indices per test user (the ranking candidates)."""
    negatives: dict[int, list[int]] = {}
    for u in test_users:
        seen = user_seen.get(u, set())
        # Can't sample more unseen items than exist; cap to avoid an infinite loop
        # on tiny corpora. Real datasets have far more items than `n`.
        target = min(n, num_items - len(seen))
        sampled: set[int] = set()
        while len(sampled) < target:
            cand = int(rng.integers(0, num_items))
            if cand not in seen:
                sampled.add(cand)
        negatives[u] = list(sampled)
    return negatives


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
@dataclass
class PreparedData:
    num_users: int
    num_items: int
    train_pairs: np.ndarray                 # shape (P, 2): [user_idx, item_idx]
    user_seen: dict[int, set[int]]
    test_items: dict[int, int]              # user_idx -> held-out item_idx
    eval_negatives: dict[int, list[int]]    # user_idx -> [item_idx, ...]
    movies: dict[int, dict]                 # item_idx -> {movie_id, title, genres}
    user_id_to_idx: dict[int, int] = field(default_factory=dict)
    item_id_to_idx: dict[int, int] = field(default_factory=dict)
    # Chronologically-ordered training item indices per user (for sequential models).
    user_sequences: dict[int, list[int]] = field(default_factory=dict)


def prepare(ratings: pd.DataFrame, movies: pd.DataFrame, settings) -> PreparedData:
    """Build everything training/eval needs from raw ratings + movie metadata."""
    user2idx, item2idx = build_index_maps(ratings)
    train_df, test_df = leave_one_out_split(ratings, settings.min_positive_rating)
    user_seen = build_user_seen(ratings, user2idx, item2idx)

    train_pairs = np.array(
        [(user2idx[u], item2idx[i]) for u, i in zip(train_df["user_id"], train_df["item_id"])],
        dtype=np.int64,
    ).reshape(-1, 2)

    # train_df is sorted by (user, timestamp), so appending preserves chronology.
    user_sequences: dict[int, list[int]] = {}
    for u, i in zip(train_df["user_id"], train_df["item_id"]):
        user_sequences.setdefault(user2idx[u], []).append(item2idx[i])

    test_items = {
        user2idx[u]: item2idx[i]
        for u, i in zip(test_df["user_id"], test_df["item_id"])
    }

    rng = np.random.default_rng(settings.seed)
    eval_negatives = sample_eval_negatives(
        user_seen, list(test_items.keys()), len(item2idx),
        settings.num_eval_negatives, rng,
    )

    movie_meta = {
        item2idx[row.item_id]: {
            "movie_id": int(row.item_id),
            "title": str(row.title),
            "genres": list(row.genres),
        }
        for row in movies.itertuples(index=False)
        if row.item_id in item2idx
    }

    return PreparedData(
        num_users=len(user2idx),
        num_items=len(item2idx),
        train_pairs=train_pairs,
        user_seen=user_seen,
        test_items=test_items,
        eval_negatives=eval_negatives,
        movies=movie_meta,
        user_id_to_idx=user2idx,
        item_id_to_idx=item2idx,
        user_sequences=user_sequences,
    )


def load_and_prepare(settings) -> PreparedData:
    """Download (if needed), load, and preprocess the configured dataset."""
    dataset_dir = download_movielens(settings.data_dir, settings.dataset)
    ratings = load_ratings(dataset_dir, settings.dataset)
    movies = load_movies(dataset_dir, settings.dataset)
    return prepare(ratings, movies, settings)

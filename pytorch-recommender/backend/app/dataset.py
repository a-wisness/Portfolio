"""Training dataset with per-epoch negative sampling.

For implicit feedback, each observed (user, item) interaction is a positive
(label 1). We pair every positive with ``num_negatives`` items the user has not
interacted with (label 0). Negatives are resampled each epoch via ``resample()``
so the model sees fresh negatives over the course of training.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class NegativeSampler:
    """Amortized-O(1) negative draws with a refillable numpy buffer.

    ``uniform`` draws items uniformly; ``popularity`` draws proportional to
    item frequency^beta — popular items make harder negatives (more likely to
    look like false positives), which can sharpen the decision boundary.
    """

    def __init__(
        self,
        num_items: int,
        rng: np.random.Generator,
        strategy: str = "uniform",
        item_counts: np.ndarray | None = None,
        beta: float = 0.75,
        buffer: int = 100_000,
    ) -> None:
        self.num_items = num_items
        self.rng = rng
        self.strategy = strategy
        self.buffer = buffer
        self._probs: np.ndarray | None = None
        if strategy == "popularity":
            if item_counts is None:
                raise ValueError("popularity sampling requires item_counts")
            weights = np.power(item_counts.astype(np.float64) + 1e-9, beta)
            self._probs = weights / weights.sum()
        elif strategy != "uniform":
            raise ValueError(f"Unknown negative_sampling strategy {strategy!r}")
        self._pool: np.ndarray = np.empty(0, dtype=np.int64)
        self._i = 0

    def _refill(self) -> None:
        if self._probs is None:
            self._pool = self.rng.integers(0, self.num_items, size=self.buffer)
        else:
            self._pool = self.rng.choice(self.num_items, size=self.buffer, p=self._probs)
        self._i = 0

    def draw(self) -> int:
        if self._i >= len(self._pool):
            self._refill()
        v = int(self._pool[self._i])
        self._i += 1
        return v


class InteractionDataset(Dataset):
    def __init__(
        self,
        train_pairs: np.ndarray,           # (P, 2) of [user_idx, item_idx]
        user_seen: dict[int, set[int]],
        num_items: int,
        num_negatives: int,
        seed: int = 42,
        sampling: str = "uniform",
        neg_pop_beta: float = 0.75,
    ) -> None:
        self.train_pairs = train_pairs
        self.user_seen = user_seen
        self.num_items = num_items
        self.num_negatives = num_negatives
        self._rng = np.random.default_rng(seed)
        counts = None
        if sampling == "popularity":
            counts = np.bincount(train_pairs[:, 1], minlength=num_items)
        self._sampler = NegativeSampler(
            num_items, self._rng, sampling, counts, neg_pop_beta
        )
        self.users: np.ndarray = np.empty(0, dtype=np.int64)
        self.items: np.ndarray = np.empty(0, dtype=np.int64)
        self.labels: np.ndarray = np.empty(0, dtype=np.float32)
        self.resample()

    def resample(self) -> None:
        """Rebuild the (user, item, label) arrays with fresh negatives."""
        n_pos = len(self.train_pairs)
        block = 1 + self.num_negatives
        users = np.empty(n_pos * block, dtype=np.int64)
        items = np.empty(n_pos * block, dtype=np.int64)
        labels = np.zeros(n_pos * block, dtype=np.float32)

        for p, (u, pos_i) in enumerate(self.train_pairs):
            base = p * block
            seen = self.user_seen.get(int(u), set())
            users[base : base + block] = u
            items[base] = pos_i
            labels[base] = 1.0
            for k in range(1, block):
                neg = self._sampler.draw()
                while neg in seen:
                    neg = self._sampler.draw()
                items[base + k] = neg
        self.users, self.items, self.labels = users, items, labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.users[idx], dtype=torch.long),
            torch.tensor(self.items[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


class BPRDataset(Dataset):
    """Pairwise (user, positive_item, negative_item) triples for BPR.

    Bayesian Personalized Ranking optimizes the *order* of a positive above a
    sampled negative for the same user, rather than calibrated probabilities.
    One fresh negative is drawn per positive each epoch via ``resample()``.
    """

    def __init__(
        self,
        train_pairs: np.ndarray,
        user_seen: dict[int, set[int]],
        num_items: int,
        seed: int = 42,
    ) -> None:
        self.train_pairs = train_pairs
        self.user_seen = user_seen
        self.num_items = num_items
        self._rng = np.random.default_rng(seed)
        self.neg_items: np.ndarray = np.empty(len(train_pairs), dtype=np.int64)
        self.resample()

    def resample(self) -> None:
        for p, (u, _pos) in enumerate(self.train_pairs):
            seen = self.user_seen.get(int(u), set())
            neg = int(self._rng.integers(0, self.num_items))
            while neg in seen:
                neg = int(self._rng.integers(0, self.num_items))
            self.neg_items[p] = neg

    def __len__(self) -> int:
        return len(self.train_pairs)

    def __getitem__(self, idx: int):
        user, pos = self.train_pairs[idx]
        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(pos, dtype=torch.long),
            torch.tensor(self.neg_items[idx], dtype=torch.long),
        )

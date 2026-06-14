"""Non-neural baselines for the benchmark.

A recommender that can't beat "just show everyone the most popular items" isn't
earning its complexity, so the popularity baseline is the reference point the
learned models must clear.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .data import PreparedData
from .evaluation import ScoreFn


class PopularityScorer:
    """Ranks items by how often they appear in the training interactions.

    Non-personalized: every user gets the same ranking. This is the floor the
    collaborative-filtering models are measured against.
    """

    def __init__(self, data: PreparedData) -> None:
        counts = np.bincount(data.train_pairs[:, 1], minlength=data.num_items)
        self.popularity = counts.astype(np.float64)

    def score_fn(self) -> ScoreFn:
        def fn(user: int, candidates: Sequence[int]) -> list[float]:
            return [float(self.popularity[i]) for i in candidates]

        return fn

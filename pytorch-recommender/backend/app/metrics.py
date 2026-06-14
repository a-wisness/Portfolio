"""Ranking metrics for the leave-one-out evaluation protocol.

Given a list of candidate items ranked best-first and the single held-out target
item, compute:

  * Hit Ratio @ K (a.k.a. Recall@K with one relevant item): 1 if the target is
    in the top K, else 0.
  * NDCG @ K: 1 / log2(rank + 2) if the target is in the top K (rank is 0-based),
    else 0.

These are pure functions over rankings so they can be unit-tested directly.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def hit_ratio_at_k(ranked_items: Sequence[int], target: int, k: int) -> float:
    return 1.0 if target in list(ranked_items)[:k] else 0.0


def ndcg_at_k(ranked_items: Sequence[int], target: int, k: int) -> float:
    top = list(ranked_items)[:k]
    if target not in top:
        return 0.0
    rank = top.index(target)  # 0-based
    return 1.0 / math.log2(rank + 2)


def mean_metrics(
    rankings: dict[int, Sequence[int]],
    targets: dict[int, int],
    k: int,
) -> dict[str, float]:
    """Average HR@K and NDCG@K over all evaluated users."""
    if not rankings:
        return {f"hr@{k}": 0.0, f"ndcg@{k}": 0.0}
    hrs, ndcgs = [], []
    for user, ranked in rankings.items():
        target = targets[user]
        hrs.append(hit_ratio_at_k(ranked, target, k))
        ndcgs.append(ndcg_at_k(ranked, target, k))
    return {
        f"hr@{k}": sum(hrs) / len(hrs),
        f"ndcg@{k}": sum(ndcgs) / len(ndcgs),
    }

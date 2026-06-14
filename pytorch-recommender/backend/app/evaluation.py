"""Model-agnostic leave-one-out ranking evaluation.

Every model (the popularity baseline, MF, GMF, MLP, NeuMF) is evaluated through
the same protocol: for each user, rank the held-out item against its sampled
negatives and average HR@K / NDCG@K. Models differ only in how they *score*
candidates, so the harness takes a ``score_fn`` rather than a concrete model.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from .data import PreparedData
from .metrics import mean_metrics

ScoreFn = Callable[[int, Sequence[int]], Sequence[float]]


def evaluate_ranking(score_fn: ScoreFn, data: PreparedData, k: int) -> dict[str, float]:
    rankings: dict[int, list[int]] = {}
    for user, target in data.test_items.items():
        candidates = [target] + data.eval_negatives[user]
        scores = score_fn(user, candidates)
        order = sorted(range(len(candidates)), key=lambda j: scores[j], reverse=True)
        rankings[user] = [candidates[j] for j in order]
    return mean_metrics(rankings, data.test_items, k)


def torch_score_fn(model: torch.nn.Module, device: torch.device) -> ScoreFn:
    """Adapt a PyTorch model's ``forward(user, item)`` into a ScoreFn."""

    @torch.no_grad()
    def fn(user: int, candidates: Sequence[int]) -> list[float]:
        model.eval()
        u = torch.full((len(candidates),), user, dtype=torch.long, device=device)
        i = torch.tensor(list(candidates), dtype=torch.long, device=device)
        return model(u, i).cpu().tolist()

    return fn

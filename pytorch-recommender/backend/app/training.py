"""Training harness shared by the CLI (`train.py`) and the benchmark.

`train_model` trains any registered model under either objective and returns the
best model (restored to its best-NDCG epoch), the per-epoch history, and the
best metrics — so both single-model training and multi-model benchmarking go
through one code path.
"""

from __future__ import annotations

import copy
import random
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .baselines import PopularityScorer
from .data import PreparedData
from .dataset import BPRDataset, InteractionDataset
from .evaluation import evaluate_ranking, torch_score_fn
from .model import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _train_bce_epoch(model, loader, optimizer, device) -> float:
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.train()
    total, count = 0.0, 0
    for users, items, labels in loader:
        users, items, labels = users.to(device), items.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(users, items), labels)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(labels)
        count += len(labels)
    return total / max(count, 1)


def _train_bpr_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total, count = 0.0, 0
    for users, pos, neg in loader:
        users, pos, neg = users.to(device), pos.to(device), neg.to(device)
        optimizer.zero_grad()
        diff = model(users, pos) - model(users, neg)
        loss = -F.logsigmoid(diff).mean()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(users)
        count += len(users)
    return total / max(count, 1)


def train_model(
    name: str,
    data: PreparedData,
    settings,
    objective: str = "bce",
    epochs: int | None = None,
    verbose: bool = False,
) -> tuple[torch.nn.Module | None, list[dict], dict[str, float]]:
    """Train one model. Returns (best_model, history, best_metrics).

    ``name="popularity"`` is a non-trained baseline: it's evaluated once and
    returns ``(None, single-entry history, metrics)``.
    """
    k = settings.top_k

    if name == "popularity":
        scorer = PopularityScorer(data)
        metrics = evaluate_ranking(scorer.score_fn(), data, k)
        return None, [{"epoch": 0, "loss": 0.0, **metrics}], metrics

    epochs = epochs or settings.epochs
    device = get_device()
    model = build_model(name, data.num_users, data.num_items, settings).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)

    if objective == "bce":
        dataset = InteractionDataset(
            data.train_pairs, data.user_seen, data.num_items,
            settings.num_negatives, settings.seed,
            sampling=settings.negative_sampling, neg_pop_beta=settings.neg_pop_beta,
        )
        epoch_fn = _train_bce_epoch
    elif objective == "bpr":
        dataset = BPRDataset(data.train_pairs, data.user_seen, data.num_items, settings.seed)
        epoch_fn = _train_bpr_epoch
    else:
        raise ValueError(f"Unknown objective {objective!r}. Choose 'bce' or 'bpr'.")

    loader = DataLoader(dataset, batch_size=settings.batch_size, shuffle=True)

    history: list[dict] = []
    best_ndcg = -1.0
    best_metrics: dict[str, float] = {}
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        dataset.resample()
        loss = epoch_fn(model, loader, optimizer, device)
        metrics = evaluate_ranking(torch_score_fn(model, device), data, k)
        history.append({"epoch": epoch, "loss": loss, **metrics})
        if metrics[f"ndcg@{k}"] >= best_ndcg:
            best_ndcg = metrics[f"ndcg@{k}"]
            best_metrics = metrics
            best_state = copy.deepcopy(model.state_dict())
        if verbose:
            print(
                f"[{name}/{objective}] epoch {epoch:2d}/{epochs}  loss={loss:.4f}  "
                f"HR@{k}={metrics[f'hr@{k}']:.4f}  NDCG@{k}={metrics[f'ndcg@{k}']:.4f}"
            )

    model.load_state_dict(best_state)  # restore best-by-NDCG weights
    return model, history, best_metrics

"""SASRec — self-attention sequential recommender (Kang & McAuley, 2018).

A *sequential* model: instead of a static user embedding, it encodes each user's
chronologically-ordered interaction history with causal self-attention and
predicts the next item. It is evaluated with the same leave-one-out next-item
protocol as the other models (rank the held-out last item against sampled
negatives), so its HR@K / NDCG@K drop straight into the benchmark.

Scope: this is an *experiment* — trained and evaluated here and added to the
benchmark report. The served model remains NeuMF.

Implementation notes:
  * Item ids are shifted by +1 so index 0 is a padding token.
  * Sequences are right-padded; the representation used for scoring is the
    hidden state at the last *real* position (so no fully-masked query rows).
  * Causal masking ensures position t attends only to positions <= t.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .data import PreparedData
from .evaluation import ScoreFn, evaluate_ranking


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class _SASBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask, key_padding_mask):
        a, _ = self.attn(
            x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.ln1(x + a)
        x = self.ln2(x + self.ff(x))
        return x


class SASRec(nn.Module):
    def __init__(
        self,
        num_items: int,
        max_len: int = 50,
        dim: int = 64,
        num_heads: int = 2,
        num_blocks: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.max_len = max_len
        self.dim = dim
        # +1 for the padding token at index 0.
        self.item_emb = nn.Embedding(num_items + 1, dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([_SASBlock(dim, num_heads, dropout) for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(dim)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.normal_(self.pos_emb.weight, std=0.01)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()  # keep pad embedding at zero

    def seq_repr(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (B, L) of shifted item ids (0 = pad) -> hidden states (B, L, D)."""
        b, length = seq.shape
        positions = torch.arange(length, device=seq.device).unsqueeze(0).expand(b, length)
        x = self.dropout(self.item_emb(seq) + self.pos_emb(positions))
        # Bool masks (True = disallowed) for both, matching key_padding_mask's dtype.
        causal = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=seq.device), diagonal=1
        )
        pad_mask = seq == 0
        for block in self.blocks:
            x = block(x, attn_mask=causal, key_padding_mask=pad_mask)
        return self.norm(x)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        return self.seq_repr(seq)

    def score(self, repr_vec: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """repr_vec: (..., D); item_ids: (..., N) shifted -> scores (..., N)."""
        emb = self.item_emb(item_ids)
        return (emb * repr_vec.unsqueeze(-2)).sum(dim=-1)


# --------------------------------------------------------------------------- #
# Sequence dataset (next-item prediction with per-position negatives)
# --------------------------------------------------------------------------- #
class SequenceDataset(Dataset):
    """Right-padded (input, positive-target, length) with resampled negatives.

    For a training sequence [i0, i1, ..., i_{m-1}], inputs are [i0..i_{m-2}] and
    positive targets are [i1..i_{m-1}] (predict the next item at each step).
    Users with fewer than 2 training items contribute no transitions.
    """

    def __init__(self, sequences: list[list[int]], num_items: int, max_len: int, seed: int = 42):
        self.num_items = num_items
        self.max_len = max_len
        self._rng = np.random.default_rng(seed)
        self.inputs: list[np.ndarray] = []
        self.targets: list[np.ndarray] = []
        for seq in sequences:
            if len(seq) < 2:
                continue
            trimmed = seq[-(max_len + 1):]
            inp = np.array(trimmed[:-1], dtype=np.int64) + 1   # shift ids (+1)
            tgt = np.array(trimmed[1:], dtype=np.int64) + 1
            self.inputs.append(self._pad(inp))
            self.targets.append(self._pad(tgt))
        self.neg = [np.zeros(max_len, dtype=np.int64) for _ in self.inputs]
        self.resample()

    def _pad(self, arr: np.ndarray) -> np.ndarray:
        out = np.zeros(self.max_len, dtype=np.int64)
        out[: len(arr)] = arr[: self.max_len]
        return out

    def resample(self) -> None:
        for n, tgt in enumerate(self.targets):
            neg = np.zeros(self.max_len, dtype=np.int64)
            for j in range(self.max_len):
                if tgt[j] == 0:
                    continue
                neg[j] = int(self._rng.integers(1, self.num_items + 1))  # shifted id
            self.neg[n] = neg

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.inputs[idx], dtype=torch.long),
            torch.tensor(self.targets[idx], dtype=torch.long),
            torch.tensor(self.neg[idx], dtype=torch.long),
        )


# --------------------------------------------------------------------------- #
# Scoring + evaluation
# --------------------------------------------------------------------------- #
def sasrec_score_fn(model: SASRec, data: PreparedData, max_len: int, device: torch.device) -> ScoreFn:
    """Score candidates for the leave-one-out eval using each user's history."""

    @torch.no_grad()
    def fn(user: int, candidates: list[int]) -> list[float]:
        model.eval()
        history = data.user_sequences.get(user, [])[-max_len:]
        seq = np.zeros(max_len, dtype=np.int64)
        if history:
            shifted = np.array(history, dtype=np.int64) + 1
            seq[: len(shifted)] = shifted
            last_idx = len(shifted) - 1
        else:
            last_idx = 0
        seq_t = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
        repr_last = model.seq_repr(seq_t)[0, last_idx]                  # (D,)
        cand = torch.tensor([c + 1 for c in candidates], dtype=torch.long, device=device)
        return model.score(repr_last, cand).cpu().tolist()

    return fn


def _train_epoch(model, loader, optimizer, device) -> float:
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    model.train()
    total, count = 0.0, 0
    for inp, pos, neg in loader:
        inp, pos, neg = inp.to(device), pos.to(device), neg.to(device)
        repr_seq = model.seq_repr(inp)                       # (B, L, D)
        pos_logits = (model.item_emb(pos) * repr_seq).sum(-1)  # (B, L)
        neg_logits = (model.item_emb(neg) * repr_seq).sum(-1)
        mask = (pos != 0).float()  # 0 = pad; only backprop through real positions
        loss_pos = loss_fn(pos_logits, torch.ones_like(pos_logits))
        loss_neg = loss_fn(neg_logits, torch.zeros_like(neg_logits))
        loss = ((loss_pos + loss_neg) * mask).sum() / mask.sum().clamp(min=1.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(inp)
        count += len(inp)
    return total / max(count, 1)


def train_sasrec(
    data: PreparedData, settings, epochs: int | None = None, verbose: bool = False
) -> tuple[SASRec, list[dict], dict[str, float]]:
    """Train SASRec; returns (best_model, history, best_metrics)."""
    from .training import get_device  # local import avoids a cycle

    epochs = epochs or settings.epochs
    device = get_device()
    k = settings.top_k

    model = SASRec(
        data.num_items, settings.sas_max_len, settings.sas_dim,
        settings.sas_heads, settings.sas_blocks, settings.sas_dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)

    sequences = list(data.user_sequences.values())
    dataset = SequenceDataset(sequences, data.num_items, settings.sas_max_len, settings.seed)
    loader = DataLoader(dataset, batch_size=settings.batch_size, shuffle=True)

    best_ndcg, best_metrics = -1.0, {}
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        dataset.resample()
        loss = _train_epoch(model, loader, optimizer, device)
        metrics = evaluate_ranking(sasrec_score_fn(model, data, settings.sas_max_len, device), data, k)
        history.append({"epoch": epoch, "loss": loss, **metrics})
        if metrics[f"ndcg@{k}"] >= best_ndcg:
            best_ndcg = metrics[f"ndcg@{k}"]
            best_metrics = metrics
            best_state = copy.deepcopy(model.state_dict())
        if verbose:
            print(f"[sasrec] epoch {epoch:2d}/{epochs}  loss={loss:.4f}  "
                  f"HR@{k}={metrics[f'hr@{k}']:.4f}  NDCG@{k}={metrics[f'ndcg@{k}']:.4f}")

    model.load_state_dict(best_state)
    return model, history, best_metrics

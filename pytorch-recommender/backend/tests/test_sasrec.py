"""Tests for the SASRec sequential model, dataset, and training."""

import torch

from app.config import settings
from app.sasrec import SASRec, SequenceDataset, train_sasrec


def test_seq_repr_shape():
    model = SASRec(num_items=20, max_len=5, dim=8, num_heads=2, num_blocks=1, dropout=0.0)
    seq = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
    out = model.seq_repr(seq)
    assert out.shape == (2, 5, 8)
    assert torch.isfinite(out).all()


def test_causal_masking_ignores_future():
    model = SASRec(num_items=20, max_len=5, dim=8, num_heads=2, num_blocks=1, dropout=0.0)
    model.eval()
    seq1 = torch.tensor([[1, 2, 3, 0, 0]])
    seq2 = torch.tensor([[1, 2, 9, 0, 0]])  # differs only at position 2
    with torch.no_grad():
        r1 = model.seq_repr(seq1)
        r2 = model.seq_repr(seq2)
    # Positions 0 and 1 cannot attend to position 2, so they must be identical.
    assert torch.allclose(r1[:, :2], r2[:, :2], atol=1e-6)
    # Position 2 itself does change.
    assert not torch.allclose(r1[:, 2], r2[:, 2], atol=1e-6)


def test_score_shape():
    model = SASRec(num_items=20, max_len=5, dim=8)
    repr_vec = torch.randn(8)
    cand = torch.tensor([1, 2, 3, 4])
    assert model.score(repr_vec, cand).shape == (4,)


def test_sequence_dataset_skips_short_and_shifts():
    ds = SequenceDataset([[5, 6, 7], [9]], num_items=20, max_len=10, seed=0)
    assert len(ds) == 1  # the length-1 sequence contributes no transition
    inp, tgt, neg = ds[0]
    # ids shifted by +1: input [6,7], target [7,8]
    assert inp[:2].tolist() == [6, 7]
    assert tgt[:2].tolist() == [7, 8]
    assert inp[2:].sum().item() == 0  # right-padded
    # negatives present where target is real, zero where padded
    assert neg[0].item() != 0 and neg[1].item() != 0
    assert neg[2:].sum().item() == 0
    # negatives are valid shifted ids
    assert 1 <= neg[0].item() <= 20


def test_train_sasrec_runs(tiny_data):
    model, history, metrics = train_sasrec(tiny_data, settings, epochs=2)
    assert isinstance(model, SASRec)
    assert len(history) == 2
    k = settings.top_k
    assert {f"hr@{k}", f"ndcg@{k}"} <= metrics.keys()

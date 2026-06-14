"""Tests for the NeuMF model and the negative-sampling dataset."""

import numpy as np
import torch

from app.dataset import InteractionDataset
from app.model import NeuMF


def test_forward_output_shape_and_finite():
    model = NeuMF(num_users=10, num_items=15, gmf_dim=8, mlp_layers=(16, 8, 4))
    users = torch.tensor([0, 1, 2, 3])
    items = torch.tensor([5, 6, 7, 8])
    out = model(users, items)
    assert out.shape == (4,)
    assert torch.isfinite(out).all()


def test_item_embedding_matrix_shape():
    model = NeuMF(num_users=10, num_items=15, gmf_dim=8, mlp_layers=(16, 8))
    emb = model.item_embedding_matrix()
    # gmf_dim (8) + mlp_embed_dim (16//2 = 8) = 16
    assert emb.shape == (15, 16)


def test_odd_first_mlp_layer_rejected():
    try:
        NeuMF(5, 5, mlp_layers=(7, 3))
    except ValueError:
        return
    raise AssertionError("expected ValueError for odd mlp_layers[0]")


def test_dataset_negative_sampling_shapes_and_labels():
    train_pairs = np.array([[0, 1], [1, 2]], dtype=np.int64)
    user_seen = {0: {1}, 1: {2}}
    ds = InteractionDataset(train_pairs, user_seen, num_items=10, num_negatives=3, seed=0)
    # 2 positives * (1 + 3) = 8 rows
    assert len(ds) == 8
    # Each block of 4 starts with a positive (label 1) then 3 negatives (label 0)
    assert ds.labels[0] == 1.0 and ds.labels[4] == 1.0
    assert ds.labels[1:4].sum() == 0.0
    # Sampled negatives must not be items the user has already seen
    for row in range(len(ds)):
        if ds.labels[row] == 0.0:
            assert ds.items[row] not in user_seen[int(ds.users[row])]


def test_dataset_resample_changes_negatives():
    train_pairs = np.array([[0, 5]], dtype=np.int64)
    ds = InteractionDataset(train_pairs, {0: set()}, num_items=1000, num_negatives=5, seed=1)
    first = ds.items.copy()
    ds.resample()
    # With 1000 items and fresh RNG draws, the negatives should differ.
    assert not np.array_equal(first, ds.items)


def test_training_reduces_loss():
    """A few optimization steps on a fixed batch should lower the BCE loss."""
    torch.manual_seed(0)
    model = NeuMF(num_users=20, num_items=30, gmf_dim=8, mlp_layers=(16, 8))
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    users = torch.randint(0, 20, (64,))
    items = torch.randint(0, 30, (64,))
    labels = (torch.rand(64) > 0.5).float()

    first = loss_fn(model(users, items), labels).item()
    for _ in range(50):
        opt.zero_grad()
        loss = loss_fn(model(users, items), labels)
        loss.backward()
        opt.step()
    assert loss.item() < first

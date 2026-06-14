"""Tests for the baseline model architectures and the factory."""

import pytest
import torch

from app.config import settings
from app.model import GMF, MLP, MatrixFactorization, NeuMF, build_model


@pytest.mark.parametrize("name,cls", [
    ("mf", MatrixFactorization),
    ("gmf", GMF),
    ("mlp", MLP),
    ("neumf", NeuMF),
])
def test_build_model_types_and_forward(name, cls):
    model = build_model(name, num_users=10, num_items=15, settings=settings)
    assert isinstance(model, cls)
    users = torch.tensor([0, 1, 2])
    items = torch.tensor([4, 5, 6])
    out = model(users, items)
    assert out.shape == (3,)
    assert torch.isfinite(out).all()


def test_build_model_unknown_raises():
    with pytest.raises(ValueError):
        build_model("transformer", 5, 5, settings)


def test_item_embedding_matrix_shapes():
    assert MatrixFactorization(10, 15, dim=8).item_embedding_matrix().shape == (15, 8)
    assert GMF(10, 15, dim=8).item_embedding_matrix().shape == (15, 8)
    # MLP embed dim = mlp_layers[0] // 2 = 8
    assert MLP(10, 15, mlp_layers=(16, 8)).item_embedding_matrix().shape == (15, 8)
    # NeuMF concatenates GMF (8) + MLP embed (8) = 16
    assert NeuMF(10, 15, gmf_dim=8, mlp_layers=(16, 8)).item_embedding_matrix().shape == (15, 16)


def test_mf_includes_bias_terms():
    mf = MatrixFactorization(5, 5, dim=4)
    # Bias-only contribution: zero embeddings would still produce the global bias.
    with torch.no_grad():
        mf.user_emb.weight.zero_()
        mf.item_emb.weight.zero_()
        mf.item_bias.weight.fill_(2.0)
        mf.user_bias.weight.fill_(1.0)
        mf.global_bias.fill_(0.5)
    out = mf(torch.tensor([0]), torch.tensor([0]))
    assert torch.allclose(out, torch.tensor([3.5]))  # 1.0 + 2.0 + 0.5

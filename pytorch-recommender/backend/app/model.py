"""Recommender model architectures (PyTorch).

All models expose the same interface — ``forward(user, item) -> logits`` of
shape (batch,) and ``item_embedding_matrix()`` — so the training harness and the
evaluation code can treat them uniformly:

  * MatrixFactorization — biased dot product of user/item embeddings (the
    classic CF baseline).
  * GMF  — Generalized Matrix Factorization: element-wise user⊗item embedding
    followed by a learned linear output (NeuMF's GMF tower, standalone).
  * MLP  — concatenated user/item embeddings through a multi-layer perceptron.
  * NeuMF — fuses independent GMF and MLP towers (He et al., 2017). This is the
    model served by the API.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _mlp_stack(input_size: int, layers: tuple[int, ...], dropout: float) -> nn.Sequential:
    modules: list[nn.Module] = []
    in_size = input_size
    for out_size in layers:
        modules.append(nn.Linear(in_size, out_size))
        modules.append(nn.ReLU())
        if dropout > 0:
            modules.append(nn.Dropout(dropout))
        in_size = out_size
    return nn.Sequential(*modules)


class MatrixFactorization(nn.Module):
    """Biased matrix factorization: <p_u, q_i> + b_u + b_i + b."""

    def __init__(self, num_users: int, num_items: int, dim: int = 32, **_: object) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(num_users, dim)
        self.item_emb = nn.Embedding(num_items, dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        dot = (self.user_emb(user) * self.item_emb(item)).sum(dim=-1)
        return dot + self.user_bias(user).squeeze(-1) + self.item_bias(item).squeeze(-1) + self.global_bias

    @torch.no_grad()
    def item_embedding_matrix(self) -> torch.Tensor:
        return self.item_emb.weight.detach()


class GMF(nn.Module):
    """Generalized Matrix Factorization tower with a learned output layer."""

    def __init__(self, num_users: int, num_items: int, dim: int = 32, **_: object) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(num_users, dim)
        self.item_emb = nn.Embedding(num_items, dim)
        self.head = nn.Linear(dim, 1)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        return self.head(self.user_emb(user) * self.item_emb(item)).squeeze(-1)

    @torch.no_grad()
    def item_embedding_matrix(self) -> torch.Tensor:
        return self.item_emb.weight.detach()


class MLP(nn.Module):
    """Concatenated user/item embeddings through a multi-layer perceptron."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        mlp_layers: tuple[int, ...] = (64, 32, 16, 8),
        dropout: float = 0.0,
        **_: object,
    ) -> None:
        super().__init__()
        if mlp_layers[0] % 2 != 0:
            raise ValueError("mlp_layers[0] must be even (it is 2 * embedding dim).")
        embed_dim = mlp_layers[0] // 2
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)
        self.mlp = _mlp_stack(mlp_layers[0], mlp_layers[1:], dropout)
        self.head = nn.Linear(mlp_layers[-1], 1)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.user_emb(user), self.item_emb(item)], dim=-1)
        return self.head(self.mlp(x)).squeeze(-1)

    @torch.no_grad()
    def item_embedding_matrix(self) -> torch.Tensor:
        return self.item_emb.weight.detach()


class NeuMF(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        gmf_dim: int = 32,
        mlp_layers: tuple[int, ...] = (64, 32, 16, 8),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if mlp_layers[0] % 2 != 0:
            raise ValueError("mlp_layers[0] must be even (it is 2 * embedding dim).")
        mlp_embed_dim = mlp_layers[0] // 2

        # GMF tower
        self.gmf_user = nn.Embedding(num_users, gmf_dim)
        self.gmf_item = nn.Embedding(num_items, gmf_dim)

        # MLP tower
        self.mlp_user = nn.Embedding(num_users, mlp_embed_dim)
        self.mlp_item = nn.Embedding(num_items, mlp_embed_dim)

        mlp_modules: list[nn.Module] = []
        in_size = mlp_layers[0]
        for out_size in mlp_layers[1:]:
            mlp_modules.append(nn.Linear(in_size, out_size))
            mlp_modules.append(nn.ReLU())
            if dropout > 0:
                mlp_modules.append(nn.Dropout(dropout))
            in_size = out_size
        self.mlp = nn.Sequential(*mlp_modules)

        # Fusion head: concat(GMF output, last MLP output) -> 1 logit
        self.head = nn.Linear(gmf_dim + mlp_layers[-1], 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for emb in (self.gmf_user, self.gmf_item, self.mlp_user, self.mlp_item):
            nn.init.normal_(emb.weight, std=0.01)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape (batch,) for each (user, item) pair."""
        gmf = self.gmf_user(user) * self.gmf_item(item)
        mlp_in = torch.cat([self.mlp_user(user), self.mlp_item(item)], dim=-1)
        mlp_out = self.mlp(mlp_in)
        fused = torch.cat([gmf, mlp_out], dim=-1)
        return self.head(fused).squeeze(-1)

    @torch.no_grad()
    def item_embedding_matrix(self) -> torch.Tensor:
        """Per-item embeddings (concat of GMF + MLP item tables).

        Used for content-free item similarity and cold-start recommendations.
        Shape: (num_items, gmf_dim + mlp_embed_dim).
        """
        return torch.cat([self.gmf_item.weight, self.mlp_item.weight], dim=-1).detach()


def build_model(name: str, num_users: int, num_items: int, settings) -> nn.Module:
    """Construct a model by name using the configured hyperparameters."""
    name = name.lower()
    if name == "mf":
        return MatrixFactorization(num_users, num_items, dim=settings.gmf_dim)
    if name == "gmf":
        return GMF(num_users, num_items, dim=settings.gmf_dim)
    if name == "mlp":
        return MLP(num_users, num_items, mlp_layers=settings.mlp_layers, dropout=settings.dropout)
    if name == "neumf":
        return NeuMF(
            num_users, num_items,
            settings.gmf_dim, settings.mlp_layers, settings.dropout,
        )
    raise ValueError(f"Unknown model {name!r}. Choose from: mf, gmf, mlp, neumf.")

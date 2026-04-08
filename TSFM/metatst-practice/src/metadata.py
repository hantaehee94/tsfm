from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn


class MetadataEncoder(nn.Module):
    """Encodes categorical and real metadata into a shared dense vector.

    The paper's central idea is that metadata can improve forecasting if we
    expose it to the model in a structured way. This module turns mixed-type
    metadata into one vector that the forecaster can condition on.
    """

    def __init__(
        self,
        categorical_cardinalities: Iterable[int],
        num_real_features: int,
        embedding_dim: int = 16,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        cardinalities: List[int] = list(categorical_cardinalities)
        # Each categorical field gets its own embedding table.
        self.categorical_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, embedding_dim) for cardinality in cardinalities]
        )

        # Concatenate embedded categorical features with real-valued metadata,
        # then project them into the model hidden size.
        input_dim = embedding_dim * len(cardinalities) + num_real_features
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        metadata_categorical: torch.Tensor,
        metadata_real: torch.Tensor,
    ) -> torch.Tensor:
        embedded = []
        for idx, embedding in enumerate(self.categorical_embeddings):
            embedded.append(embedding(metadata_categorical[:, idx]))

        parts = embedded + [metadata_real]
        merged = torch.cat(parts, dim=-1)
        return self.projection(merged)

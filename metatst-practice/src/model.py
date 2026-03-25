from __future__ import annotations

import torch
from torch import nn

from src.metadata import MetadataEncoder


class MetaForecastTransformer(nn.Module):
    """Transformer forecaster conditioned on metadata.

    We first map the raw past values into token embeddings, then add two kinds
    of context:
    - positional information
    - metadata-derived conditioning vector
    """

    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        categorical_cardinalities: list[int],
        num_real_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length

        # Each scalar value in the history becomes one transformer token.
        self.value_projection = nn.Linear(1, d_model)
        self.metadata_encoder = MetadataEncoder(
            categorical_cardinalities=categorical_cardinalities,
            num_real_features=num_real_features,
            hidden_dim=d_model,
        )
        # Learned positional embedding keeps the model aware of time order.
        self.position_embedding = nn.Parameter(torch.randn(1, context_length, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, prediction_length),
        )

    def forward(
        self,
        past_values: torch.Tensor,
        metadata_categorical: torch.Tensor,
        metadata_real: torch.Tensor,
    ) -> torch.Tensor:
        values = past_values.unsqueeze(-1)
        value_tokens = self.value_projection(values)
        metadata_context = self.metadata_encoder(
            metadata_categorical=metadata_categorical,
            metadata_real=metadata_real,
        ).unsqueeze(1)

        # Broadcast metadata across the whole context window so every timestep
        # can attend while being aware of the series descriptor.
        tokens = value_tokens + self.position_embedding
        tokens = tokens + metadata_context
        encoded = self.encoder(tokens)

        # A simple starter choice: summarize the encoded sequence with the last
        # token and map it to the full future horizon.
        pooled = encoded[:, -1, :]
        return self.head(pooled)

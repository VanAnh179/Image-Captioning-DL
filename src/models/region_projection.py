"""Trainable region projection shared by the two decoder architectures."""

import torch
from torch import nn

from src.data.region_features import FEATURE_DIM, NUM_REGIONS


class RegionProjection(nn.Module):
    """Linear + LayerNorm + learned grid positions, from [B,49,2048] to [B,49,d]."""

    def __init__(self, d_model, dropout, feature_dim=FEATURE_DIM, num_regions=NUM_REGIONS):
        super().__init__()
        self.linear = nn.Linear(feature_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.position = nn.Embedding(num_regions, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, regions):
        expected = (self.position.num_embeddings, self.linear.in_features)
        if regions.ndim != 3 or tuple(regions.shape[1:]) != expected or regions.shape[0] == 0:
            raise ValueError(f"regions must have shape [B,{expected[0]},{expected[1]}] with B > 0")
        index = torch.arange(regions.shape[1], device=regions.device)
        regions = regions.to(dtype=self.linear.weight.dtype)
        return self.dropout(self.norm(self.linear(regions)) + self.position(index))

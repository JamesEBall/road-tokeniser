"""Small Graph Attention encoder for road-segment tokens.

Architecture:
    [linear projection F → d]
        ↓
    L × [GATConv(d, d, heads=4) + residual + LayerNorm]
        ↓
    embedding of dim d per token

Edge types (along-road vs shares-junction) are folded into a single message-
passing graph; in Phase C we may switch to a HeteroGraph Transformer if the
loss curves want it. For 12 k tokens this is plenty.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class RoadEncoder(nn.Module):
    """Graph Attention encoder producing per-token embeddings."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 128,
        num_layers: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            # GATConv with concat=False averages heads → output is hidden_dim
            self.layers.append(
                GATConv(hidden_dim, hidden_dim, heads=heads, concat=False, dropout=dropout)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.out_proj = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for conv, norm in zip(self.layers, self.norms):
            update = conv(h, edge_index)
            h = norm(h + self.dropout(F.gelu(update)))
        return self.out_proj(h)


class MaskedFeatureHead(nn.Module):
    """Reconstruction head: embedding → masked-feature prediction.

    Predicts the full input feature vector. SSL loss masks a random subset of
    nodes and computes MSE on numerical features + cross-entropy on highway
    class for those nodes.
    """

    def __init__(
        self,
        embed_dim: int,
        n_numeric: int,
        n_binary: int,
        n_highway_classes: int,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        hd = hidden_dim or embed_dim
        self.shared = nn.Sequential(nn.Linear(embed_dim, hd), nn.GELU())
        self.numeric_head = nn.Linear(hd, n_numeric)
        self.binary_head = nn.Linear(hd, n_binary)
        self.highway_head = nn.Linear(hd, n_highway_classes)

    def forward(self, embeddings: torch.Tensor):
        h = self.shared(embeddings)
        return {
            "numeric": self.numeric_head(h),
            "binary": self.binary_head(h),
            "highway": self.highway_head(h),
        }


class RoadFoundationModel(nn.Module):
    """Encoder + pretraining heads. Use `encode()` for downstream embeddings."""

    def __init__(
        self,
        in_dim: int,
        n_numeric: int,
        n_binary: int,
        n_highway: int,
        embed_dim: int = 128,
        num_layers: int = 4,
        heads: int = 4,
    ):
        super().__init__()
        self.encoder = RoadEncoder(
            in_dim=in_dim,
            hidden_dim=embed_dim,
            out_dim=embed_dim,
            num_layers=num_layers,
            heads=heads,
        )
        self.head = MaskedFeatureHead(embed_dim, n_numeric, n_binary, n_highway)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        emb = self.encode(x, edge_index)
        out = self.head(emb)
        out["embeddings"] = emb
        return out

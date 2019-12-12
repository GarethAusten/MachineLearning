"""Dot Product Model"""

import torch
import torch.nn as nn


class DotProdModel(nn.Module):
    """Torch Dot Model."""
    def __init__(self, hidden_size1, hidden_size2, hidden_size3,
                 vocab_size, dropout, pretrained=False, weights=None,
                 emb_dim=None):
        """Initialization."""
        super(DotProdModel, self).__init__()

        if pretrained:
            self.embedding = nn.Embedding(weights.size(0), weights.size(1))
            self.embedding.weight.data.copy_(weights)
            emb_dim = weights.size(1)
        else:
            self.embedding = nn.Embedding(vocab_size, emb_dim)

        self.stack = nn.ModuleList([
            nn.Linear(emb_dim, hidden_size1),
            nn.Linear(hidden_size1, hidden_size2),
            nn.Linear(hidden_size2, hidden_size3)
            ])

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.Tanh()

    def forward(self, x):
        """Forward pass."""
        embedded = torch.sum(self.embedding(x), dim=0)
        for layer in self.stack:
            output = self.dropout(self.activation(layer(embedded)))
        return output
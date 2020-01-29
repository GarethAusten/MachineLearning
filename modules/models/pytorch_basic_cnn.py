"""Basic CNN PyTorch"""
# coding=utf-8

import torch

import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """Basic 2D CNN Class."""
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes,
                 output_dim, dropout, pad_idx):
        """
        """
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim,
                                      padding_idx=pad_idx)

        self.convolutions = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(filter_size, embedding_dim))
            for filter_size in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        """Forward pass."""
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)

        output = [F.relu(convolution(embedded)).squeeze(3) for convolution in
                  self.convolutions]
        pooled_output = [F.max_pool1d(
            convolution, convolution.shape[2]).squeeze(2) for  convolution in
                  output]

        cat = self.dropout(torch.cat(pooled_output, dim=1))
        return self.fc(cat)
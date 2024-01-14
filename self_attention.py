from math import inf, sqrt

import torch
from torch import nn
from torch.nn import functional as F


class SelfAttentionHead(nn.Module):
    def __init__(self, head_size: int, n_embd: int, block_size: int):
        super().__init__()
        self.head_size = head_size
        self.n_embd = n_embd
        self.key = nn.Linear(self.n_embd, self.head_size, bias=False)
        self.query = nn.Linear(self.n_embd, self.head_size, bias=False)
        self.value = nn.Linear(self.n_embd, self.head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape

        # C must equal the n_embd of the key, query, and value vectors
        assert C == self.n_embd

        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)

        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * self.head_size ** -0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, -inf)  # B, T, T
        wei = F.softmax(wei, dim=-1)  # (B, T, T)

        # Perform weighted aggregation of the values
        v = self.value(x)  # (B, T, C)
        return wei @ v  # (B, T, head_size)


def test_self_attention():
    B = 4  # batch_size
    T = 8  # time / block_size
    C = 32  # channels / n_embd
    head_size = 16

    head = SelfAttentionHead(head_size=16, n_embd=C, block_size=T)
    x = torch.randn(B, T, C)
    y = head(x)
    assert y.shape == (B, T, head_size)

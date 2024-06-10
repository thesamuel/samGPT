from math import inf

import torch
from torch import nn
from torch.nn import functional as F
from typing_extensions import deprecated


@deprecated("For reference only; use SelfAttentionHead(n_head=1, ...)")
class SelfAttentionSingleHead(nn.Module):
    """
    Single-head self attention implementation

    Included for reference, should always use SelfAttentionHead
    """

    def __init__(self, head_size: int, n_embd: int, block_size: int):
        super().__init__()
        self.head_size = head_size
        self.n_embd = n_embd
        self.key = nn.Linear(self.n_embd, self.head_size, bias=False)
        self.query = nn.Linear(self.n_embd, self.head_size, bias=False)
        self.value = nn.Linear(self.n_embd, self.head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape

        # C must equal the n_embd of the key, query, and value vectors
        assert C == self.n_embd

        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)

        # Compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * self.head_size**-0.5
        )  # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, -inf)  # B, T, T
        wei = F.softmax(wei, dim=-1)  # (B, T, T)

        # Perform weighted aggregation of the values
        v = self.value(x)  # (B, T, head_size)
        return wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, head_size)


class SelfAttentionHead(nn.Module):
    def __init__(self, n_embd: int, block_size: int, n_head: int = 1):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.attention = nn.Linear(self.n_embd, 3 * self.n_embd)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        # TODO: add more shape assertions throughout
        B, T, C = x.shape

        # C must equal the n_embd of the key, query, and value vectors
        assert C == self.n_embd

        q, k, v = (
            self.attention(x)  # (B, T, C, n_embd)
            .view((B, T, 3, self.n_head, self.head_size))
            .movedim(2, 0)
            .transpose(2, 3)
        )  # 3 x (B, n_head, T, head_size)

        # Compute attention scores ("affinities"), scaled by head size
        # (B, n_head, T, head_size) @ (B, n_head, head_size, T) -> (B, n_head, T, T)
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, -inf)  # (B, n_head, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, n_head, T, T)

        # (B, n_head, T, T) @ (B, n_head, T, head_size) -> (B, n_head, T, head_size)
        out = wei @ v

        # Transpose the heads to the final dimensions and flatten them
        return out.transpose(1, 2).contiguous().view(B, T, C)


def test_self_attention():
    # Set seed to make the test reproducible
    torch.manual_seed(42)

    B = 4  # batch_size
    T = 8  # time / block_size
    C = 32  # channels / n_embd

    head = SelfAttentionHead(n_embd=C, block_size=T, n_head=1)
    x = torch.ones(B, T, C)
    y = head(x)
    assert y.shape == (B, T, C)

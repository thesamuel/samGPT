from math import inf

import numpy as np
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
    # Buffers
    tril: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor

    def __init__(
        self,
        n_embd: int,
        block_size: int,
        n_head: int = 1,
        cache: bool = False,
        # TODO: pass this in from LanguageModel
        max_cache_batch_size: int = 1,
    ):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.attention = nn.Linear(self.n_embd, 3 * self.n_embd)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.cache = cache
        if self.cache:
            cache_shape = (
                max_cache_batch_size,
                self.n_head,
                block_size,
                self.head_size,
            )
            self.register_buffer("k_cache", torch.zeros(cache_shape))
            self.register_buffer("v_cache", torch.zeros(cache_shape))

    def forward(self, x: torch.Tensor, input_pos: int = 0) -> torch.Tensor:
        # TODO: add more shape assertions throughout
        # TODO: try using einops!

        B, T, C = x.shape

        # C must equal the n_embd of the key, query, and value vectors
        assert C == self.n_embd

        if self.cache and input_pos > 0:
            x = x[:, -1:, :]
            T_attn = 1
        else:
            T_attn = T

        q, k, v = (
            self.attention(x)  # (B, T, C, n_embd)
            .view((B, T_attn, 3, self.n_head, self.head_size))
            .movedim(2, 0)
            .transpose(2, 3)
        )  # 3 x (B, n_head, T, head_size)

        if self.cache:
            # TODO: there's probably some code cleanup we could do here:
            #  look at https://github.com/pytorch/pytorch/blob/df43d58/benchmarks/gpt_fast/mixtral_moe_model.py#L210
            if input_pos > 0:
                self.k_cache[:, :, T - 1] = k.squeeze(dim=2)
                self.v_cache[:, :, T - 1] = v.squeeze(dim=2)
            else:
                self.k_cache[:, :, :T] = k
                self.v_cache[:, :, :T] = v
            k = self.k_cache[:, :, :T]
            v = self.v_cache[:, :, :T]

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


def test_self_attention_cached():
    # Set seed to make the test reproducible
    torch.manual_seed(42)

    # TODO: test when B < max_cache_batch_size
    B = 4  # batch_size
    T = 8  # time / block_size
    C = 32  # channels / n_embd

    # TODO: try B + 1
    # TODO: try n_head = 4
    # TODO: ensure that cache=False == cache=True
    head = SelfAttentionHead(
        n_embd=C, block_size=T, n_head=1, cache=True, max_cache_batch_size=B
    )
    x = torch.ones(B, T, C)

    with torch.no_grad():
        # This is the first case
        _ = head(x[:, : T - 1, :], input_pos=0)
        # This should have a cache hit
        y = head(x[:, :T, :], input_pos=1)
        assert y.shape == (B, T, C)

        # Disable caching
        head.cache = False
        y_no_cache = head(x)
        np.testing.assert_allclose(
            actual=y.numpy(), desired=y_no_cache.numpy(), rtol=1e-5
        )

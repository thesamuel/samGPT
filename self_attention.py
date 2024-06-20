from math import inf
from typing import Optional

import numpy as np
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from typing_extensions import deprecated


@deprecated("For reference only; use SelfAttentionHead(n_head=1, ...)")
class SelfAttentionSingleHead(nn.Module):
    """
    Single-head self attention implementation

    Included for reference, should always use SelfAttentionHead
    """

    def __init__(
        self,
        n_embd: int,
        block_size: int,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
    ):
        super().__init__()
        self.n_embd = n_embd

        # Generally the dimensions are the same throughout (at least in Attention is All You Need)
        d_k = d_k or self.n_embd
        d_v = d_v or self.n_embd

        self.key = nn.Linear(self.n_embd, d_k, bias=False)
        self.query = nn.Linear(self.n_embd, d_k, bias=False)
        self.value = nn.Linear(self.n_embd, d_v, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        """
        Time Complexity
        ------------
        All the below assumes that d_k == d_v == n_embd
        Time:
        - q, k, and v embeddings: O(B * T * n_embd^2) [FACT CHECK THIS]
        - Attention: O(B * T^2 * n_embd)
            - q@k: O(B * T^2 * n_embd)
            - wei@v: O(B * T^2 * n_embd)
            - softmax, masked fill: O(B * T^2)

        Space Complexity
        ------------
        TODO: complete this
            - How much space is used for...
                - activations?
                    - Assuming T > n_embd: O(B * T^2) [FACT CHECK THIS]
                - weights?
                    # - O(n_embd^2) for weights, gradients, optimizer for q, k, v [FACT CHECK THIS]
                - gradients, optimizer states (during training)?
                - Are there optimizations for this?

        TODO: open questions
            - What are some previous papers on attention to read?
            - What about follow-ups? Flash attention, etc.
        """
        B, T, C = x.shape

        # C must equal the n_embd of the key, query, and value vectors
        # Generally, C == d_k == d_v as well, but it's not required
        assert C == self.n_embd

        # Complexity:
        # - time: O(B * T * C * head_size),
        # O(B * T * C) for activations
        # weights: O(n_embd * n_embd)

        k = self.key(x)  # (B, T, d_k)
        q = self.query(x)  # (B, T, d_k)

        # Everything after this is "Attention", as defined in "Attention is All You Need":
        # Compute attention scores ("affinities")
        # More simply: q dot k gives us a weighting over the value embeddings so that we can do a weighted average
        # Note: we need to transpose the last two dimensions to do the tensor-multiplication (think of it as B matmuls)
        scale = self.key.out_features**-0.5  # 1/sqrt(d_k)
        wei = q @ k.mT * scale  # (B, T, d_k) @ (B, d_k, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, -inf)  # B, T, T
        wei = F.softmax(wei, dim=-1)  # (B, T, T)

        # Perform weighted aggregation of the values
        v = self.value(x)  # (B, T, d_v)
        # Note that if d_v != n_embd, we must project this back to n_embd before re-applying self-attention
        return wei @ v  # (B, T, T) @ (B, T, d_v) -> (B, T, d_v)


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
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_embd = n_embd  # Just used for shape checks
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

    def get_cache(
        self, input_pos: int, T: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: there's probably some code cleanup we could do here:
        #  look at https://github.com/pytorch/pytorch/blob/df43d58/benchmarks/gpt_fast/mixtral_moe_model.py#L210
        if input_pos == 0:
            # Create cache
            self.k_cache[:, :, :T] = k
            self.v_cache[:, :, :T] = v
        else:
            # Update cache
            self.k_cache[:, :, T - 1] = k.squeeze(dim=2)
            self.v_cache[:, :, T - 1] = v.squeeze(dim=2)

        # Return updated, cached values
        return self.k_cache[:, :, :T], self.v_cache[:, :, :T]

    def forward(self, x: torch.Tensor, input_pos: int = 0) -> torch.Tensor:
        B, T, C = x.shape
        assert C == self.n_embd

        T_attn = T
        if self.cache and input_pos > 0:
            # If caching is enabled, and we've already processed the sequence up to this point, we just need to
            # process the newest token
            x = x[:, -1:, :]
            T_attn = 1

        attn = self.attention(x)  # (B, T, 3 * C)
        q, k, v = rearrange(
            attn,
            "b t (qkv nh hs) -> qkv b nh t hs",
            qkv=3,
            t=T_attn,
            nh=self.n_head,
            hs=self.head_size,
        )  # q, k, v: (B, n_head, T, head_size)

        if self.cache:
            # Use cached k, v
            k, v = self.get_cache(input_pos, T, k, v)

        # Compute attention scores ("affinities"), scaled by head size
        # (B, n_head, T, head_size) @ (B, n_head, head_size, T) -> (B, n_head, T, T)
        wei = q @ k.mT * self.head_size**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, -inf)  # (B, n_head, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, n_head, T, T)

        # (B, n_head, T, T) @ (B, n_head, T, head_size) -> (B, n_head, T, head_size)
        out = wei @ v

        # Transpose the heads to the final dimensions and flatten them
        return rearrange(out, "b nh t hs -> b t (nh hs)")


def test_single_head_self_attention():
    # Set seed to make the test reproducible
    torch.manual_seed(42)

    B = 4  # batch_size
    T = 8  # time / block_size
    C = 32  # channels / n_embd

    head = SelfAttentionSingleHead(n_embd=C, block_size=T)
    x = torch.ones(B, T, C)
    y = head(x)
    assert y.shape == (B, T, C)


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

from typing import Optional

from self_attention import SelfAttentionHead
from torch import nn
from torch.nn import functional as F
import torch


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        n_embd: int,
        block_size: int,
        n_head: int,
        ff_width: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.sa = nn.Sequential(
            SelfAttentionHead(n_embd, block_size, n_head),
            nn.Linear(n_embd, n_embd),
            nn.Dropout(dropout),
        )
        self.ff = nn.Sequential(
            nn.Linear(n_embd, ff_width * n_embd),
            nn.ReLU(),
            nn.Linear(ff_width * n_embd, n_embd),
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Create residual connections
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class LanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int,
        n_head: int = 1,
        n_block: int = 3,
        ff_width: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(self.block_size, n_embd)
        self.blocks = nn.Sequential(
            *(
                SelfAttentionBlock(
                    n_embd, block_size, n_head, ff_width, dropout=dropout
                )
                for _ in range(n_block)
            )
        )
        self.ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.register_buffer("arange", torch.arange(self.block_size))

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = x.shape

        tok_emb = self.token_embedding_table(x)  # (B, T, C)
        # TODO: do we need to use a device here?
        pos_emb = self.position_embedding_table(self.arange[:T])

        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # Apply one head of self attention (B, T, C)
        x = self.ln(x)  # Apply final layer norm
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, x: torch.Tensor, max_new_tokens: int):
        # TODO: should this be with model.eval / torch.no_grad / torch.inference_mode?
        out = x
        for _ in range(max_new_tokens):
            # Call the model with the previous block size of tokens
            x_cond = out[:, -self.block_size :]
            logits, _ = self(x_cond)

            # The final row of the model output corresponds to the next token, so we slice to just that row
            logits = logits[:, -1, :]

            # Then, we sample from the distribution and append it to our output
            probs = F.softmax(logits, dim=-1)
            # TODO: try different sampling approaches
            next_token = torch.multinomial(probs, num_samples=1)
            out = torch.cat((out, next_token), dim=1)
        return out

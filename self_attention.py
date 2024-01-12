from math import inf

import torch
from torch import nn
from torch.nn import functional as F

B = 4
T = 8
C = 32
x = torch.randn(B, T, C)

head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

k = key(x)
q = query(x)
wei = q @ k.transpose(-2, -1)  # (B, T, head_size) @ (B, 16, T) -> (B, T, T)

tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, -inf)
wei = F.softmax(wei, dim=-1)

v = value(x)
out = wei @ v

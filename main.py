import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask=None
    ):
        d_k = query.size(-1)
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            causal_mask = torch.tril(torch.ones_like(scores))
            scores = scores.masked_fill(causal_mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1) @ value

        return attn

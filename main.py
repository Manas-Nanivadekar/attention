import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


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


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        assert (
            self.head_dim * num_heads == d_model
        ), "Embedding dimension must be divisible by number of heads"

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query = self.w_q(query)
        key = self.w_k(key)
        value = self.w_v(value)

        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(
            1, 2
        )

        x = self.attention(query, key, value, mask=mask)

        x = x.transpose(2, 1).contiguous().view(batch_size, -1, self.d_model)

        return self.w_o(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, layer, N, vocab_size):
        super().__init__()
        self.embd = nn.Embedding(vocab_size, layer.d_model)
        self.pe = PositionalEncoding(layer.d_model, dropout=layer.dropout.p)
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(self, x, mask):
        x = self.embd(x) * math.sqrt(x.size(-1))
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.src_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        src_attn = self.src_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout(src_attn))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

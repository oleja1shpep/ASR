import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout_prob: float = 0.1):
        super().__init__()

        self.layernorm = nn.LayerNorm(hidden_dim)
        # add pos encoding in future
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layernorm(x)
        x, _ = self.attention(x, x, x, need_weights=False)
        return self.dropout(x)

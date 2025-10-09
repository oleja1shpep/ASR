import torch
import torch.nn.functional as F
from torch import nn


class RoPE(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.base = 10000
        self.emb_dim = emb_dim

        self.theta = self.base ** (-2 * torch.arange(emb_dim // 2) / emb_dim)
        self.idxs = torch.arange(emb_dim) // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : tensor of shape (B, heads, seq_len, d_head)
        """
        seq_len = x.shape[2]
        seq_idx = torch.arange(seq_len)

        theta_matrix = torch.einsum("m,d->md", seq_idx, self.theta).to(x.device)

        cos = theta_matrix.cos()[None, None, :, :]
        sin = theta_matrix.sin()[None, None, :, :]

        x1, x2 = x[..., ::2], x[..., 1::2]

        output = torch.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        return output.flatten(-2)


class RoPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = 512, num_heads: int = 8):
        super().__init__()
        assert (
            d_model % num_heads == 0
        ), "The dimension should be devisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = int(d_model / num_heads)

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.rope_q = RoPE(self.d_head)
        self.rope_k = RoPE(self.d_head)

        self.linear = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = value.size(0)
        seq_len = value.size(1)

        query = (
            self.W_q(query)
            .view(batch_size, seq_len, self.num_heads, self.d_head)
            .transpose(1, 2)
        )
        key = (
            self.W_k(key)
            .view(batch_size, seq_len, self.num_heads, self.d_head)
            .transpose(1, 2)
        )
        value = (
            self.W_v(value)
            .view(batch_size, seq_len, self.num_heads, self.d_head)
            .transpose(1, 2)
        )

        rope_query = self.rope_q(query)
        rope_key = self.rope_k(key)

        key_padding_mask_expanded = torch.logical_not(
            key_padding_mask.view(batch_size, 1, 1, seq_len).expand(
                -1, self.num_heads, -1, -1
            )
        )

        # mask should be (bs, num_heads, 1, seq_len)
        output = F.scaled_dot_product_attention(
            rope_query, rope_key, value, key_padding_mask_expanded
        )
        output = output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.linear(output)


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout_prob: float = 0.1):
        super().__init__()

        self.layernorm = nn.LayerNorm(hidden_dim)
        self.attention = RoPEMultiHeadAttention(hidden_dim, num_heads)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        x = self.layernorm(x)
        x = self.attention(x, x, x, key_padding_mask=key_padding_mask)
        return self.dropout(x)

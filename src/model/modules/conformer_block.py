import torch
from torch import nn

from src.model.modules.attention import MultiHeadAttention
from src.model.modules.convolution import ConvModule
from src.model.modules.ffn import FeedForward


class ConformerBlock(nn.Module):
    def __init__(
        self,
        block_dim: int,
        ffn_droput_prob: float = 0.1,
        ffn_expansion_factor: int = 4,
        ffn_residual_factor: float = 0.5,
        num_heads: int = 8,
        attn_dropout_prob: float = 0.1,
        conv_kernel_size: int = 3,
        conv_expansion_factor: int = 2,
        conv_dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.ffn1 = FeedForward(
            hidden_dim=block_dim,
            expansion_factor=ffn_expansion_factor,
            dropout_prob=ffn_droput_prob,
        )
        self.ffn_residual_factor = ffn_residual_factor
        self.attention = MultiHeadAttention(
            hidden_dim=block_dim, num_heads=num_heads, dropout_prob=attn_dropout_prob
        )
        self.conv = ConvModule(
            in_channels=block_dim,
            expansion_factor=conv_expansion_factor,
            kernel_size=conv_kernel_size,
            dropout_prob=conv_dropout_prob,
        )
        self.ffn2 = FeedForward(
            hidden_dim=block_dim,
            expansion_factor=ffn_expansion_factor,
            dropout_prob=ffn_droput_prob,
        )
        self.layernorm = nn.LayerNorm(block_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ffn1(x) * self.ffn_residual_factor
        x = x + self.attention(x)
        x = x + self.conv(x)
        x = x + self.ffn2(x) * self.ffn_residual_factor
        return self.layernorm(x)

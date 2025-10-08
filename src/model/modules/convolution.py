import torch
from torch import nn

from src.model.modules.activation import SwishActivation


class ConvModule(nn.Module):
    """
    Conformer convulotion block
    """

    def __init__(
        self, in_channels, expansion_factor: int = 2, kernel_size=3, dropout_prob=0.1
    ):
        super().__init__()

        self.layernorm = nn.LayerNorm(in_channels)

        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels, in_channels * expansion_factor, kernel_size=1
            ),  # pointwize
            nn.GLU(dim=1),
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                groups=in_channels,
            ),  # depthwize
            nn.BatchNorm1d(in_channels),
            SwishActivation(),
            nn.Conv1d(in_channels, in_channels, kernel_size=1),  # pointwize
            nn.Dropout(dropout_prob),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layernorm(x).transpose(1, 2)
        return self.layers(x).transpose(1, 2)

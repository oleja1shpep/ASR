import torch
from torch import nn

from src.model.modules.activation import SwishActivation


class FeedForward(nn.Module):
    """
    Feed Forward Network
    """

    def __init__(
        self, hidden_dim: int, expansion_factor: int = 4, dropout_prob: float = 0.1
    ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * expansion_factor),
            SwishActivation(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim * expansion_factor, hidden_dim),
            nn.Dropout(dropout_prob),
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)

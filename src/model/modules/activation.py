import torch
from torch import nn


class SwishActivation(nn.Module):
    """
    Swish Activation Module
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x.sigmoid()

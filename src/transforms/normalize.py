import torch
from torch import nn


class Normalize1D(nn.Module):
    def __init__(self, mean, std, *args, **kwargs):
        super().__init__()

        self.mean = mean
        self.std = std

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """
        data: tensor of shape (batch, mel_dim, time)
        """

        return (data - self.mean) / self.std

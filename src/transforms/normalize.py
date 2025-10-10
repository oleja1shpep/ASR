import torch
import torchvision.transforms.functional as F
from torch import nn
from torchvision.transforms.v2 import Normalize


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

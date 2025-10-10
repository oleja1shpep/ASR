import numpy as np
import torch
from torch import nn


class Noise(nn.Module):
    def __init__(self, std=0.05, p=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noiser = torch.distributions.Normal(0, std)
        self.p = p

    def __call__(self, audio: torch.Tensor):
        if np.random.rand() < self.p:
            audio = audio + self.noiser.sample(audio.size()).to(audio.device)
        return audio

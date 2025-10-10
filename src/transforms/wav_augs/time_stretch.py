import numpy as np
import torch
from librosa.effects import time_stretch
from torch import nn


class TimeStretch(nn.Module):
    def __init__(self, stretch_factor=[0.95, 1.05], p=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.stretch_factor = stretch_factor

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        if np.random.rand() < self.p:
            factor = np.random.uniform(*self.stretch_factor)
            audio = time_stretch(audio.squeeze().numpy(), rate=factor)
            audio = torch.from_numpy(audio)[None, :]
        return audio

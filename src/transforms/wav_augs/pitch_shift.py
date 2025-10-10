import numpy as np
import torch
from torch import nn
from torch_audiomentations import PitchShift as Pitch


class PitchShift(nn.Module):
    def __init__(self, pitch_factor=[-3, 3], p=0.5, sample_rate=16000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pither = Pitch(*pitch_factor, p=p, sample_rate=sample_rate)

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        x = audio.unsqueeze(1)
        return self.pither(x).squeeze(1)

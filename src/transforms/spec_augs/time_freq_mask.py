import torch
from torch import nn
from torchaudio.transforms import FrequencyMasking, TimeMasking


class TimeFreqMasking(nn.Module):
    def __init__(self, freq_size=20, time_size=40, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._aug = nn.Sequential(
            FrequencyMasking(freq_size),
            TimeMasking(time_size),
        )

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Tensor of shape (mel_dim, )
        """
        return self._aug(spec)

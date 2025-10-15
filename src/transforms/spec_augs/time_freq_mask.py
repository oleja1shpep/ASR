import torch
from torch import nn
from torchaudio.transforms import FrequencyMasking, TimeMasking


class TimeFreqMasking(nn.Module):
    def __init__(
        self,
        freq_size=20,
        time_size=40,
        freq_repeats=1,
        time_repeats=1,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.augs = nn.ModuleList(
            [FrequencyMasking(freq_size) for _ in range(freq_repeats)]
            + [TimeMasking(time_size) for _ in range(time_repeats)]
        )

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Tensor of shape (mel_dim, time)
        """
        for aug in self.augs:
            spec = aug(spec)
        return spec

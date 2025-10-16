import torch_audiomentations
from torch import Tensor, nn


class Impulse(nn.Module):
    def __init__(self, p=0.5, sample_rate=16000):
        super().__init__()
        self._aug = torch_audiomentations.ApplyImpulseResponse(
            ir_paths="./impulses/MIT_Survey/", p=p, sample_rate=sample_rate
        )

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)

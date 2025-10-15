import torch
from torch import nn


class Conv1dSubsampling(nn.Module):
    """
    Convolutional 1D subsampling (to 1/3 length)
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=3, padding=1),
            nn.ReLU(),
        )

    def forward(
        self, inputs: torch.Tensor, input_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        inputs : tensor of shape (batch, dim, time)
        """
        outputs = self.layers(inputs)
        output_lengths = (input_lengths) // 3
        return outputs.transpose(1, 2), output_lengths


class Conv2dSubsampling(nn.Module):
    """
    Convolutional 2D subsampling (to 1/2 length)
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
        )

    def forward(
        self, inputs: torch.Tensor, input_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.layers(inputs.unsqueeze(1))
        batch_size, channels, subsampled_lengths, sumsampled_dim = outputs.size()

        outputs = outputs.permute(0, 2, 1, 3)
        outputs = outputs.contiguous().view(
            batch_size, subsampled_lengths, channels * sumsampled_dim
        )

        output_lengths = (input_lengths - 1) // 2

        return outputs, output_lengths

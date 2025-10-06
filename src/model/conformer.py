import torch
from torch import nn

from src.model.modules import ConformerBlock, Conv1dSubsampling, Conv2dSubampling


class Conformer(nn.Module):
    """
    Conformer
    """

    def __init__(
        self,
        n_feats: int,
        n_layers: int,
        n_tokens: int,
        conformer_block_dim: int = 512,
        initial_dropout_prob: float = 0.1,
        ffn_droput_prob: float = 0.1,
        ffn_expansion_factor: int = 4,
        ffn_residual_factor: float = 0.5,
        attn_num_heads: int = 8,
        attn_dropout_prob: float = 0.1,
        conv_kernel_size: int = 5,
        conv_expansion_factor: int = 2,
        conv_dropout_prob: float = 0.1,
    ):
        """
        Args:
            n_feats (int): number of input features.
            n_tokens (int): number of tokens in the vocabulary.
            fc_hidden (int): number of hidden features.
        """
        super().__init__()

        self.conv_subsampling = Conv2dSubampling(1, conformer_block_dim)
        self.linear = nn.Linear(
            conformer_block_dim * ((n_feats - 1) // 2), conformer_block_dim
        )
        self.dropout1 = nn.Dropout(initial_dropout_prob)
        self.conformer_blocks = nn.ModuleList(
            [
                ConformerBlock(
                    block_dim=conformer_block_dim,
                    ffn_droput_prob=ffn_droput_prob,
                    ffn_expansion_factor=ffn_expansion_factor,
                    ffn_residual_factor=ffn_residual_factor,
                    num_heads=attn_num_heads,
                    attn_dropout_prob=attn_dropout_prob,
                    conv_kernel_size=conv_kernel_size,
                    conv_expansion_factor=conv_expansion_factor,
                    conv_dropout_prob=conv_dropout_prob,
                )
                for _ in range(n_layers)
            ]
        )

        self.fc = nn.Linear(conformer_block_dim, n_tokens, bias=False)

    def forward(self, spectrogram, spectrogram_length, **batch):
        """
        Model forward method.

        Args:
            spectrogram (Tensor): input spectrogram of shape (batch, time, dim)
            spectrogram_length (Tensor): spectrogram original lengths.
        Returns:
            output (dict): output dict containing log_probs and
                transformed lengths.
        """
        subsampled, output_lengths = self.conv_subsampling(
            spectrogram.transpose(1, 2), spectrogram_length
        )
        outputs = self.dropout1(self.linear(subsampled))

        for block in self.conformer_blocks:
            outputs = block(outputs)

        outputs = self.fc(outputs)
        outputs = nn.functional.log_softmax(outputs, dim=-1)
        return {"log_probs": outputs, "log_probs_length": output_lengths}

    def transform_input_lengths(self, input_lengths):
        """
        As the network may compress the Time dimension, we need to know
        what are the new temporal lengths after compression.

        Args:
            input_lengths (Tensor): old input lengths
        Returns:
            output_lengths (Tensor): new temporal lengths
        """
        return input_lengths  # we don't reduce time dimension here

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info

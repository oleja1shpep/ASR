import torch
from torch import Tensor
from torch.nn import CTCLoss


class CTCLossWrapper(CTCLoss):
    def forward(
        self, log_probs, log_probs_length, text_encoded, text_encoded_length, **batch
    ) -> Tensor:
        log_probs_t = torch.transpose(log_probs, 0, 1)

        loss = super().forward(
            log_probs=log_probs_t,
            targets=text_encoded,
            input_lengths=log_probs_length,
            target_lengths=text_encoded_length,
        )
        if loss.isinf().any():
            import pdb

            pdb.set_trace()

        # if torch.any(log_probs == torch.inf):
        #     print("BLYAT log_probs")
        # if torch.any(log_probs_length == torch.inf):
        #     print("BLYAT log_probs_length")
        # if torch.any(log_probs_length == 0):
        #     print("BLYAT log_probs_length")
        # if torch.any(text_encoded == torch.nan):
        #     print("BLYAT text_encoded")
        # if torch.any(text_encoded_length == torch.nan):
        #     print("BLYAT text_encoded_length")

        return {"loss": loss}

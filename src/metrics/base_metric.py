from abc import abstractmethod

from torch import Tensor


class BaseMetric:
    """
    Base class for all metrics
    """

    def __init__(self, name=None, *args, **kwargs):
        """
        Args:
            name (str | None): metric name to use in logger and writer.
        """
        self.name = name if name is not None else type(self).__name__

    def precompute_preds(self, log_probs: Tensor, log_probs_length: Tensor, **batch):
        beam_search_texts = self.text_encoder.ctc_beam_search(
            log_probs.cpu(), log_probs_length
        )
        return beam_search_texts

    @abstractmethod
    def __call__(self, **batch):
        """
        Defines metric calculation logic for a given batch.
        Can use external functions (like TorchMetrics) or custom ones.
        """
        raise NotImplementedError()

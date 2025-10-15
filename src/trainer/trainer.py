from pathlib import Path

import pandas as pd

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.metrics.utils import calc_cer, calc_wer
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        preds = None
        # precompute for beam search
        for met in metric_funcs:
            if met.name.endswith("(Beam_Search)"):
                preds = met.precompute_preds(**batch)
                break

        for met in metric_funcs:
            if met.name.endswith("(Beam_Search)"):
                metrics.update(met.name, met(preds=preds, **batch))
            else:
                metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_spectrogram(**batch)
            self.log_audio(log_ori=True, **batch)
        else:
            # Log Stuff
            self.log_spectrogram(**batch)
            self.log_audio(**batch)
            self.log_predictions(**batch)

    def log_audio(self, audio, ori_audio, log_ori=False, **batch):
        self.writer.add_audio("audio", audio[0], sample_rate=16000)
        if log_ori:
            self.writer.add_audio("ori_audio", ori_audio[0], sample_rate=16000)

    def log_spectrogram(self, spectrogram, **batch):
        spectrogram_for_plot = spectrogram[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("spectrogram", image)

    def log_predictions(
        self, text, log_probs, log_probs_length, audio_path, examples_to_log=10, **batch
    ):
        # Note: by improving text encoder and metrics design
        # this logging can also be improved significantly

        cpu_log_probs = log_probs.cpu()

        beam_search_texts = self.text_encoder.ctc_beam_search(
            cpu_log_probs, log_probs_length
        )

        tuples = list(zip(beam_search_texts, text, audio_path))

        rows = {}
        for beam_pred, target, audio_path in tuples[:examples_to_log]:
            target = self.text_encoder.normalize_text(target)

            beam_wer = calc_wer(target, beam_pred) * 100
            beam_cer = calc_cer(target, beam_pred) * 100

            rows[Path(audio_path).name] = {
                "target": target,
                "predictions": beam_pred,
                "wer": beam_wer,
                "cer": beam_cer,
            }
        self.writer.add_table(
            "predictions", pd.DataFrame.from_dict(rows, orient="index")
        )

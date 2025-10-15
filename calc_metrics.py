import warnings
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate

from src.metrics.tracker import MetricTracker
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="calc_metrics")
def main(config):
    text_encoder = instantiate(config.text_encoder)

    metrics = {"inference": []}
    for metric_config in config.metrics.get("inference", []):
        # use text_encoder in metrics
        metrics["inference"].append(
            instantiate(metric_config, text_encoder=text_encoder)
        )

    tracker = MetricTracker(
        *[m.name for m in metrics["inference"]],
    )

    pred_dir = ROOT_PATH / config.predictions_dir
    target_dir = ROOT_PATH / config.target_dir

    if not (target_dir.exists()):
        raise ValueError("target directory seems to be non-existing")

    targets = []
    preds = []

    for target_path in Path(target_dir).iterdir():
        with open(target_path, "r") as f:
            target = f.read().strip()
        if not (pred_dir.exists()):
            raise ValueError("predictions directory seems to be non-existing")

        pred_path = pred_dir / (target_path.name)
        with open(pred_path, "r") as f:
            pred = f.read().strip()
        targets.append(target)
        preds.append(pred)

    for met in metrics["inference"]:
        tracker.update(met.name, met(preds=preds, text=targets))
    logs = tracker.result()
    tracker.reset()
    for key, value in logs.items():
        print(f"    {key:15s}: {value}")


if __name__ == "__main__":
    main()

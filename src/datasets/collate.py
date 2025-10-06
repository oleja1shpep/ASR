import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    spectrogram = torch.nn.utils.rnn.pad_sequence(
        [item["spectrogram"].squeeze(0).transpose(0, 1) for item in dataset_items],
        batch_first=True,
    ).transpose(1, 2)
    text_encoded = torch.nn.utils.rnn.pad_sequence(
        [item["text_encoded"].squeeze(0) for item in dataset_items], batch_first=True
    )
    text_encoded_length = torch.tensor(
        [item["text_encoded"].shape[-1] for item in dataset_items], dtype=torch.int32
    )

    return dict(
        spectrogram=spectrogram,
        audio=[item["audio"] for item in dataset_items],
        text=[item["text"] for item in dataset_items],
        text_encoded=text_encoded,
        text_encoded_length=text_encoded_length,
        audio_path=[item["audio_path"] for item in dataset_items],
        spectrogram_length=torch.tensor(
            [item["spectrogram"].shape[-1] for item in dataset_items], dtype=torch.int32
        ),
    )

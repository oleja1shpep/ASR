from pathlib import Path

import torchaudio

from src.datasets.base_dataset import BaseDataset


class CustomDirAudioDataset(BaseDataset):
    def __init__(
        self,
        data_dir,
        audio_dir="audio",
        transcription_dir="transcriptions",
        *args,
        **kwargs
    ):
        data = []
        audio_dir = Path.joinpath(Path(data_dir), audio_dir)
        transcription_dir = Path.joinpath(Path(data_dir), transcription_dir)
        for path in Path(audio_dir).iterdir():
            entry = {}
            if path.suffix in [".mp3", ".wav", ".flac", ".m4a"]:
                entry["path"] = str(path)
                info = torchaudio.info(entry["path"])
                length = info.num_frames / info.sample_rate
                entry["audio_len"] = length
                if transcription_dir and Path(transcription_dir).exists():
                    transc_path = Path(transcription_dir) / (path.stem + ".txt")
                    if transc_path.exists():
                        with transc_path.open() as f:
                            entry["text"] = f.read().strip()
            if len(entry) > 0:
                data.append(entry)
        super().__init__(data, *args, **kwargs)

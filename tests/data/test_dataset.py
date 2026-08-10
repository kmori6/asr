import json
from pathlib import Path

import soundfile as sf
import torch

from asr.data import SpeechTextDataset


def test_speech_text_dataset_loads_relative_audio_path(tmp_path: Path) -> None:
    waveform = torch.tensor([0.0, 0.25, -0.25, 0.5], dtype=torch.float32)
    audio_path = tmp_path / "sample.flac"
    sf.write(audio_path, waveform.numpy(), samplerate=16_000)

    manifest_path = tmp_path / "train.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "id": "1-2-3",
                    "audio_path": audio_path.name,
                    "num_samples": waveform.numel(),
                    "sample_rate": 16_000,
                    "text": "hello world",
                }
            ]
        ),
        encoding="utf-8",
    )

    dataset = SpeechTextDataset(manifest_path)
    sample = dataset[0]

    assert len(dataset) == 1
    assert sample.utterance_id == "1-2-3"
    assert sample.sample_rate == 16_000
    assert sample.text == "hello world"
    torch.testing.assert_close(sample.waveform, waveform, atol=1e-4, rtol=0.0)

from pathlib import Path

import soundfile as sf
import torch
from torchaudio.functional import resample

from asr.data import load_audio


def test_load_audio_downmixes_and_resamples(tmp_path: Path) -> None:
    stereo_waveform = torch.tensor(
        [
            [0.0, 0.5],
            [0.5, 0.0],
            [-0.5, 0.0],
            [0.0, -0.5],
        ],
        dtype=torch.float32,
    )
    audio_path = tmp_path / "stereo.wav"
    sf.write(audio_path, stereo_waveform.numpy(), samplerate=8_000, subtype="FLOAT")

    waveform = load_audio(audio_path, sample_rate=16_000)

    assert waveform.dtype == torch.float32
    assert waveform.ndim == 1
    assert waveform.shape[0] == 8
    assert torch.isfinite(waveform).all()
    expected = resample(stereo_waveform.mean(dim=1), 8_000, 16_000)
    torch.testing.assert_close(waveform, expected)

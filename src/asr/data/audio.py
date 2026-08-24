from pathlib import Path

import soundfile as sf
import torch
from torchaudio.functional import resample


def load_audio(path: str | Path, sample_rate: int) -> torch.Tensor:
    """Load arbitrary audio as a finite mono waveform at ``sample_rate``.

    Multi-channel audio is downmixed by averaging channels. Audio whose source
    sample rate differs from the requested rate is resampled with torchaudio.

    Args:
        path: Input audio path supported by SoundFile.
        sample_rate: Positive output sample rate in Hz.

    Returns:
        CPU ``torch.float32`` waveform with shape ``(num_samples,)``.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    audio_path = Path(path).expanduser()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    waveform, source_sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    if waveform.shape[0] == 0:
        raise ValueError(f"Audio must contain at least one sample: {audio_path}")

    mono_waveform = torch.from_numpy(waveform).mean(dim=1)
    if not torch.isfinite(mono_waveform).all():
        raise ValueError(f"Audio must contain only finite samples: {audio_path}")
    if source_sample_rate != sample_rate:
        mono_waveform = resample(mono_waveform, source_sample_rate, sample_rate)
    return mono_waveform.contiguous()

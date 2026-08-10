import torch

from asr.modules.frontend import LogMelSpectrogram


def test_log_mel_spectrogram_returns_valid_feature_lengths() -> None:
    torch.manual_seed(0)
    waveforms = torch.randn(2, 1_600)
    waveforms[1, 1_200:] = 0.0
    waveform_lengths = torch.tensor([1_600, 1_200])
    frontend = LogMelSpectrogram()

    features, feature_lengths = frontend(waveforms, waveform_lengths)

    assert features.shape == (2, 7, 80)
    torch.testing.assert_close(feature_lengths, torch.tensor([7, 5]))
    assert torch.isfinite(features).all()
    torch.testing.assert_close(features[1, 5:], torch.zeros_like(features[1, 5:]))

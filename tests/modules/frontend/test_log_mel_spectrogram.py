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


def test_log_mel_spectrogram_chunk_outputs_match_full_outputs() -> None:
    torch.manual_seed(0)
    waveform = torch.randn(1, 29)
    frontend = LogMelSpectrogram(sample_rate=16, n_fft=8, hop_length=3, n_mels=4)

    full_features, full_lengths = frontend(waveform, torch.tensor([waveform.shape[1]]))

    cache = None
    chunk_features = []
    chunk_start = 0
    for chunk_length in (2, 7, 1, 9, 10):
        chunk = waveform[:, chunk_start : chunk_start + chunk_length]
        features, feature_lengths, cache = frontend.forward_chunk(
            chunk,
            torch.tensor([chunk_length]),
            cache,
        )
        chunk_features.append(features[:, : feature_lengths.item()])
        chunk_start += chunk_length

    assert cache is not None
    streaming_features = torch.cat(chunk_features, dim=1)
    torch.testing.assert_close(streaming_features, full_features)
    assert streaming_features.shape[1] == full_lengths.item()
    torch.testing.assert_close(cache.waveform_lengths, torch.tensor([5]))

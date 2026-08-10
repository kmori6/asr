import torch
import torch.nn as nn
from torchaudio.functional import melscale_fbanks


class LogMelSpectrogram(nn.Module):
    """Convert padded waveforms into streaming-compatible log-Mel features."""

    window: torch.Tensor
    mel_filters: torch.Tensor

    def __init__(
        self,
        sample_rate: int = 16_000,
        n_fft: int = 512,
        hop_length: int = 160,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: float | None = None,
    ) -> None:
        super().__init__()
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if n_fft <= 0:
            raise ValueError("n_fft must be positive")
        if hop_length <= 0:
            raise ValueError("hop_length must be positive")
        if n_mels <= 0:
            raise ValueError("n_mels must be positive")

        nyquist_frequency = sample_rate / 2
        maximum_frequency = nyquist_frequency if f_max is None else f_max
        if not 0 <= f_min < maximum_frequency <= nyquist_frequency:
            raise ValueError("frequencies must satisfy 0 <= f_min < f_max <= sample_rate / 2")

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.register_buffer("window", torch.hann_window(n_fft, periodic=True))
        self.register_buffer(
            "mel_filters",
            melscale_fbanks(
                n_freqs=n_fft // 2 + 1,
                f_min=f_min,
                f_max=maximum_frequency,
                n_mels=n_mels,
                sample_rate=sample_rate,
                norm="slaney",
                mel_scale="slaney",
            ),
        )

    def forward(self, waveforms: torch.Tensor, waveform_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute log-Mel features without using future padding.

        Args:
            waveforms: Padded mono waveforms with shape ``(batch, num_samples)``.
            waveform_lengths: Valid sample counts with shape ``(batch,)``.

        Returns:
            Log-Mel features with shape ``(batch, num_frames, n_mels)`` and valid
            feature lengths with shape ``(batch,)``. Invalid padded frames are zero.

        Note:
            Frames are not centered, so each frame only depends on samples already received.
            Computation and output use float32 for FFT and logarithm stability.
        """
        if waveforms.ndim != 2:
            raise ValueError(f"waveforms must have shape (batch, num_samples), but got {tuple(waveforms.shape)}")
        if waveform_lengths.ndim != 1 or waveform_lengths.shape[0] != waveforms.shape[0]:
            raise ValueError(
                "waveform_lengths must have shape (batch,), "
                f"but got {tuple(waveform_lengths.shape)} for batch size {waveforms.shape[0]}"
            )
        if not waveforms.is_floating_point():
            raise TypeError("waveforms must be a floating-point tensor")
        if torch.any(waveform_lengths < self.n_fft) or torch.any(waveform_lengths > waveforms.shape[1]):
            raise ValueError(f"waveform lengths must be between n_fft ({self.n_fft}) and the padded waveform length")

        waveforms = waveforms.float()
        spectrogram = torch.stft(
            waveforms,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window.to(device=waveforms.device, dtype=waveforms.dtype),
            center=False,
            normalized=False,  # False for causal
            return_complex=True,
        )  # (batch_size, n_freqs = n_fft // 2 + 1, n_frames = (n_samples - n_fft) // hop_length + 1)
        power_spectrogram = spectrogram.abs().square().transpose(1, 2)
        mel_spectrogram = power_spectrogram @ self.mel_filters.to(
            device=waveforms.device,
            dtype=power_spectrogram.dtype,
        )
        features = mel_spectrogram.clamp_min(1e-10).log()

        feature_lengths = (
            torch.div(
                waveform_lengths - self.n_fft,
                self.hop_length,
                rounding_mode="floor",
            )
            + 1
        )
        frame_indices = torch.arange(features.shape[1], device=features.device)
        valid_frames = frame_indices[None, :] < feature_lengths[:, None]
        features = features.masked_fill(~valid_frames[:, :, None], 0.0)
        return features, feature_lengths

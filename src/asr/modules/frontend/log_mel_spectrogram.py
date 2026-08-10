from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torchaudio.functional import melscale_fbanks


@dataclass(frozen=True, slots=True)
class LogMelSpectrogramCache:
    """Unprocessed waveform samples retained across streaming chunks."""

    waveforms: torch.Tensor
    waveform_lengths: torch.Tensor


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
        if hop_length > n_fft:
            raise ValueError("hop_length must be no greater than n_fft")
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
        self._validate_waveforms(waveforms, waveform_lengths)
        if torch.any(waveform_lengths < self.n_fft) or torch.any(waveform_lengths > waveforms.shape[1]):
            raise ValueError(f"waveform lengths must be between n_fft ({self.n_fft}) and the padded waveform length")

        features = self._compute_features(waveforms)
        feature_lengths = self._feature_lengths(waveform_lengths)
        return self._mask_invalid_frames(features, feature_lengths), feature_lengths

    def forward_chunk(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor,
        cache: LogMelSpectrogramCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, LogMelSpectrogramCache]:
        """
        Args:
            waveforms: Next padded waveform chunk with shape
                ``(batch, num_chunk_samples)``.
            waveform_lengths: Valid samples in the chunk with shape ``(batch,)``.
            cache: Unprocessed samples returned by the preceding call for the
                same utterances, or ``None`` for the first chunks.

        Returns:
            Newly available log-Mel frames, their valid lengths, and the samples
            beginning at the next uncomputed frame boundary. Discard the cache
            after the final chunk of each utterance.

        Note:
            Only complete ``n_fft``-sample frames are emitted. The cache keeps
            fewer than ``n_fft`` samples per utterance, so no frame is duplicated
            or padded with samples that have not arrived yet.
        """
        self._validate_waveforms(waveforms, waveform_lengths)
        if torch.any(waveform_lengths < 0) or torch.any(waveform_lengths > waveforms.shape[1]):
            raise ValueError("waveform lengths must be between 0 and the padded chunk length")

        if cache is None:
            cached_waveforms = waveforms.new_empty((waveforms.shape[0], 0))
            cached_lengths = waveform_lengths.new_zeros(waveforms.shape[0])
        else:
            self._validate_cache(cache, waveforms)
            cached_waveforms = cache.waveforms
            cached_lengths = cache.waveform_lengths

        combined_lengths = cached_lengths + waveform_lengths
        combined_waveforms = pad_sequence(
            [
                torch.cat(
                    (
                        cached_waveforms[index, : cached_lengths[index]],
                        waveforms[index, : waveform_lengths[index]],
                    )
                )
                for index in range(waveforms.shape[0])
            ],
            batch_first=True,
        )

        if combined_waveforms.shape[1] < self.n_fft:
            features = waveforms.new_empty((waveforms.shape[0], 0, self.n_mels), dtype=torch.float32)
            feature_lengths = waveform_lengths.new_zeros(waveforms.shape[0])
        else:
            features = self._compute_features(combined_waveforms)
            feature_lengths = self._feature_lengths(combined_lengths)
            features = self._mask_invalid_frames(features, feature_lengths)

        consumed_samples = feature_lengths * self.hop_length
        cached_sequences = [
            combined_waveforms[index, consumed_samples[index] : combined_lengths[index]]
            for index in range(waveforms.shape[0])
        ]
        next_cache = LogMelSpectrogramCache(
            waveforms=pad_sequence(cached_sequences, batch_first=True),
            waveform_lengths=combined_lengths - consumed_samples,
        )
        return features, feature_lengths, next_cache

    def _compute_features(self, waveforms: torch.Tensor) -> torch.Tensor:
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
        return mel_spectrogram.clamp_min(1e-10).log()

    def _feature_lengths(self, waveform_lengths: torch.Tensor) -> torch.Tensor:
        return (
            torch.div(
                waveform_lengths - self.n_fft,
                self.hop_length,
                rounding_mode="floor",
            )
            + 1
        ).clamp_min(0)

    @staticmethod
    def _mask_invalid_frames(features: torch.Tensor, feature_lengths: torch.Tensor) -> torch.Tensor:
        frame_indices = torch.arange(features.shape[1], device=features.device)
        valid_frames = frame_indices[None, :] < feature_lengths[:, None]
        return features.masked_fill(~valid_frames[:, :, None], 0.0)

    @staticmethod
    def _validate_waveforms(waveforms: torch.Tensor, waveform_lengths: torch.Tensor) -> None:
        if waveforms.ndim != 2:
            raise ValueError(f"waveforms must have shape (batch, num_samples), but got {tuple(waveforms.shape)}")
        if waveform_lengths.ndim != 1 or waveform_lengths.shape[0] != waveforms.shape[0]:
            raise ValueError(
                "waveform_lengths must have shape (batch,), "
                f"but got {tuple(waveform_lengths.shape)} for batch size {waveforms.shape[0]}"
            )
        if not waveforms.is_floating_point():
            raise TypeError("waveforms must be a floating-point tensor")
        if waveform_lengths.dtype != torch.long:
            raise TypeError("waveform_lengths must have dtype torch.long")
        if waveform_lengths.device != waveforms.device:
            raise ValueError("waveforms and waveform_lengths must be on the same device")

    def _validate_cache(self, cache: LogMelSpectrogramCache, waveforms: torch.Tensor) -> None:
        if cache.waveforms.ndim != 2 or cache.waveforms.shape[0] != waveforms.shape[0]:
            raise ValueError("cached waveforms must have shape (batch, num_cached_samples)")
        if cache.waveform_lengths.shape != (waveforms.shape[0],):
            raise ValueError("cached waveform_lengths must have shape (batch,)")
        if cache.waveform_lengths.dtype != torch.long:
            raise TypeError("cached waveform_lengths must have dtype torch.long")
        if cache.waveforms.device != waveforms.device or cache.waveform_lengths.device != waveforms.device:
            raise ValueError("waveform chunks and cache must be on the same device")
        if cache.waveforms.dtype != waveforms.dtype:
            raise TypeError("waveform chunks and cached waveforms must have the same dtype")
        if torch.any(cache.waveform_lengths < 0) or torch.any(cache.waveform_lengths >= self.n_fft):
            raise ValueError("cached waveform lengths must be between 0 and n_fft - 1")
        if torch.any(cache.waveform_lengths > cache.waveforms.shape[1]):
            raise ValueError("cached waveform lengths must not exceed the padded cache length")

import torch
import torch.nn as nn
from torchaudio.functional import melscale_fbanks


class LogMelSpectrogram(nn.Module):
    """Convert waveforms into causal log-Mel features."""

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
        """

        Args:
            waveforms (torch.Tensor): Padded mono waveforms with shape ``(batch, num_samples)``.
            waveform_lengths (torch.Tensor): Valid sample counts with shape ``(batch,)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Log-Mel features with shape ``(batch, num_frames, n_mels)``
                and the corresponding valid frame counts with shape ``(batch,)``.
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
        cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            waveforms (torch.Tensor): Next unpadded waveform chunk for one utterance with shape
                ``(1, num_chunk_samples)``.
            cache (torch.Tensor | None, optional): Samples retained from the preceding chunk
                with shape ``(1, num_cached_samples)``, or ``None`` for the first chunk.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Newly available log-Mel frames and the samples
                beginning at the next STFT frame boundary. Discard the cache after the final chunk.

        Note:
            Only complete ``n_fft``-sample frames are emitted. The cache keeps
            fewer than ``n_fft`` samples, so no frame is duplicated or padded
            with samples that have not arrived yet.
        """
        self._validate_chunk(waveforms)
        if cache is not None:
            self._validate_cache(cache, waveforms)
        buffered_waveforms = waveforms if cache is None else torch.cat((cache, waveforms), dim=1)

        if buffered_waveforms.shape[1] < self.n_fft:
            features = waveforms.new_empty((1, 0, self.n_mels), dtype=torch.float32)
        else:
            features = self._compute_features(buffered_waveforms)

        consumed_samples = features.shape[1] * self.hop_length
        next_cache = buffered_waveforms[:, consumed_samples:]
        return features, next_cache

    def _compute_features(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Compute log-Mel features without using future samples.

        Args:
            waveforms (torch.Tensor): Batched mono waveforms with shape ``(batch, num_samples)``.

        Returns:
            torch.Tensor: Log-Mel features with shape ``(batch, num_frames, n_mels)``.
        """
        # NOTE: torch.stft requires float32 for numerical stability
        waveforms = waveforms.float()
        spectrogram = torch.stft(
            waveforms,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window.to(device=waveforms.device, dtype=waveforms.dtype),
            center=False,  # Do not use future samples.
            normalized=False,
            return_complex=True,
        )  # (batch_size, n_freqs = n_fft // 2 + 1, n_frames = (n_samples - n_fft) // hop_length + 1)
        power_spectrogram = spectrogram.abs().square().transpose(1, 2)  # (batch_size, n_frames, n_freqs)
        mel_spectrogram = power_spectrogram @ self.mel_filters.to(
            device=waveforms.device,
            dtype=power_spectrogram.dtype,
        )
        return mel_spectrogram.clamp_min(1e-10).log()

    def _feature_lengths(self, waveform_lengths: torch.Tensor) -> torch.Tensor:
        """Compute the number of valid causal log-Mel frames for each waveform.

        Args:
            waveform_lengths (torch.Tensor): Valid sample counts with shape ``(batch,)``.

        Returns:
            torch.Tensor: Number of complete frames that fit inside each waveform under the
            causal ``n_fft``/``hop_length`` formulation, clamped to zero for shorter inputs.
        """
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
        """Mask out frames that fall beyond each valid feature length.

        Args:
            features (torch.Tensor): Log-Mel feature tensors with shape ``(batch, num_frames, n_mels)``.
            feature_lengths (torch.Tensor): Valid frame counts with shape ``(batch,)``.

        Returns:
            torch.Tensor: A copy of ``features`` with invalid padded frames replaced by zero.
        """
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

    @staticmethod
    def _validate_chunk(waveforms: torch.Tensor) -> None:
        if waveforms.ndim != 2 or waveforms.shape[0] != 1:
            raise ValueError(
                f"streaming waveforms must have shape (1, num_chunk_samples), but got {tuple(waveforms.shape)}"
            )
        if not waveforms.is_floating_point():
            raise TypeError("streaming waveforms must be a floating-point tensor")

    def _validate_cache(self, cache: torch.Tensor, waveforms: torch.Tensor) -> None:
        if cache.ndim != 2 or cache.shape[0] != 1 or cache.shape[1] >= self.n_fft:
            raise ValueError("cached waveforms must have shape (1, num_cached_samples) with fewer than n_fft samples")
        if cache.device != waveforms.device:
            raise ValueError("waveform chunks and cache must be on the same device")
        if cache.dtype != waveforms.dtype:
            raise TypeError("waveform chunks and cached waveforms must have the same dtype")

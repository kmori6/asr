import torch
import torch.nn as nn


class SpecAugment(nn.Module):
    """Apply frequency and time masking to log-Mel features.

    Time warping is intentionally omitted. The masking operations follow
    https://www.isca-archive.org/interspeech_2019/park19e_interspeech.pdf.
    """

    def __init__(
        self,
        num_frequency_masks: int,
        max_frequency_mask_width: int,
        num_time_masks: int,
        max_time_mask_width: int,
    ) -> None:
        super().__init__()
        if num_frequency_masks < 0 or num_time_masks < 0:
            raise ValueError("numbers of masks must be non-negative")
        if max_frequency_mask_width < 0 or max_time_mask_width < 0:
            raise ValueError("maximum mask widths must be non-negative")

        self.num_frequency_masks = num_frequency_masks
        self.max_frequency_mask_width = max_frequency_mask_width
        self.num_time_masks = num_time_masks
        self.max_time_mask_width = max_time_mask_width

    def forward(self, features: torch.Tensor, feature_lengths: torch.Tensor) -> torch.Tensor:
        """Mask valid regions of a batch of log-Mel features.

        Args:
            features: Log-Mel features with shape ``(batch, num_frames, n_mels)``.
            feature_lengths: Valid frame counts with shape ``(batch,)``.

        Returns:
            Features with independently sampled frequency and time masks. Evaluation
            mode returns the input unchanged. Shape, dtype, and device are preserved.
        """
        if features.ndim != 3:
            raise ValueError(f"features must have shape (batch, num_frames, n_mels), but got {tuple(features.shape)}")
        if feature_lengths.ndim != 1 or feature_lengths.shape[0] != features.shape[0]:
            raise ValueError(
                "feature_lengths must have shape (batch,), "
                f"but got {tuple(feature_lengths.shape)} for batch size {features.shape[0]}"
            )
        if feature_lengths.device != features.device:
            raise ValueError("features and feature_lengths must be on the same device")

        batch_size, num_frames, n_mels = features.shape
        if torch.any(feature_lengths < 0) or torch.any(feature_lengths > num_frames):
            raise ValueError(f"feature lengths must be between 0 and the padded feature length ({num_frames})")
        if self.max_frequency_mask_width > n_mels:
            raise ValueError(
                f"max_frequency_mask_width ({self.max_frequency_mask_width}) must not exceed n_mels ({n_mels})"
            )
        if not self.training:
            return features

        frequency_lengths = feature_lengths.new_full((batch_size,), n_mels)
        frequency_mask = self._sample_mask(
            lengths=frequency_lengths,
            max_length=n_mels,
            num_masks=self.num_frequency_masks,
            max_mask_width=self.max_frequency_mask_width,
        )
        time_mask = self._sample_mask(
            lengths=feature_lengths,
            max_length=num_frames,
            num_masks=self.num_time_masks,
            max_mask_width=self.max_time_mask_width,
        )

        frame_indices = torch.arange(num_frames, device=features.device)
        valid_frames = frame_indices[None, :] < feature_lengths[:, None]
        mask = valid_frames[:, :, None] & (time_mask[:, :, None] | frequency_mask[:, None, :])
        return features.masked_fill(mask, 0.0)

    @staticmethod
    def _sample_mask(
        lengths: torch.Tensor,
        max_length: int,
        num_masks: int,
        max_mask_width: int,
    ) -> torch.Tensor:
        """Sample independent contiguous masks for a batch of sequences."""
        batch_size = lengths.shape[0]
        if num_masks == 0 or max_mask_width == 0:
            return torch.zeros((batch_size, max_length), dtype=torch.bool, device=lengths.device)

        shape = (batch_size, num_masks)
        maximum_widths = lengths.clamp(max=max_mask_width)[:, None]
        widths = torch.floor(torch.rand(shape, device=lengths.device) * (maximum_widths + 1)).long()
        start_limits = lengths[:, None] - widths + 1
        starts = torch.floor(torch.rand(shape, device=lengths.device) * start_limits).long()

        positions = torch.arange(max_length, device=lengths.device)[None, None, :]
        masks = (positions >= starts[:, :, None]) & (positions < (starts + widths)[:, :, None])
        return masks.any(dim=1)

import torch
import torch.nn as nn
import torch.nn.functional as F


class FastConformerSubsampling(nn.Module):
    """Apply causal 8x convolutional subsampling to acoustic features.

    The first stage is a regular strided convolution. The second and third
    stages are depthwise-separable strided convolutions, following the
    FastConformer subsampling architecture described in
    https://arxiv.org/pdf/2305.05084.
    """

    _NUM_STAGES = 3
    _KERNEL_SIZE = 3
    _STRIDE = 2

    def __init__(self, input_size: int, output_size: int, conv_channels: int = 256) -> None:
        super().__init__()
        if input_size <= 0:
            raise ValueError("input_size must be positive")
        if output_size <= 0:
            raise ValueError("output_size must be positive")
        if conv_channels <= 0:
            raise ValueError("conv_channels must be positive")

        self.input_size = input_size
        subsampling_factor = self._STRIDE**self._NUM_STAGES
        frequency_size = (input_size + subsampling_factor - 1) // subsampling_factor

        self.input_conv = nn.Conv2d(
            in_channels=1,
            out_channels=conv_channels,
            kernel_size=self._KERNEL_SIZE,
            stride=self._STRIDE,
            padding=(0, 1),
        )
        self.depthwise_convs = nn.ModuleList(
            nn.Conv2d(
                in_channels=conv_channels,
                out_channels=conv_channels,
                kernel_size=self._KERNEL_SIZE,
                stride=self._STRIDE,
                padding=(0, 1),
                groups=conv_channels,
            )
            for _ in range(self._NUM_STAGES - 1)
        )
        self.pointwise_convs = nn.ModuleList(
            nn.Conv2d(conv_channels, conv_channels, kernel_size=1) for _ in range(self._NUM_STAGES - 1)
        )
        self.activation = nn.ReLU()
        self.output_projection = nn.Linear(conv_channels * frequency_size, output_size)

    def forward(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Subsample a padded batch of acoustic features.

        Args:
            features: Acoustic features with shape ``(batch, num_frames, input_size)``.
            feature_lengths: Valid frame counts with shape ``(batch,)``.

        Returns:
            Encoded features with shape ``(batch, ceil(num_frames / 8), output_size)``
            and valid lengths ``ceil(feature_lengths / 8)``. Invalid output frames
            are zero.

        Note:
            Time-axis padding is added only on the left. Each output therefore
            depends only on its current and previous input frames.
        """
        if features.ndim != 3:
            raise ValueError(f"features must have shape (batch, num_frames, input_size), but got {features.shape}")
        if features.shape[-1] != self.input_size:
            raise ValueError(f"expected input_size {self.input_size}, but got {features.shape[-1]}")
        if feature_lengths.ndim != 1 or feature_lengths.shape[0] != features.shape[0]:
            raise ValueError(
                "feature_lengths must have shape (batch,), "
                f"but got {tuple(feature_lengths.shape)} for batch size {features.shape[0]}"
            )
        if feature_lengths.device != features.device:
            raise ValueError("features and feature_lengths must be on the same device")
        if torch.any(feature_lengths < 0) or torch.any(feature_lengths > features.shape[1]):
            raise ValueError(f"feature lengths must be between 0 and the padded feature length ({features.shape[1]})")

        x = features.unsqueeze(1)  # (batch, 1, num_frames, input_size)
        x = self.activation(self.input_conv(self._pad_time(x)))
        for depthwise_conv, pointwise_conv in zip(self.depthwise_convs, self.pointwise_convs, strict=True):
            x = self.activation(pointwise_conv(depthwise_conv(self._pad_time(x))))

        batch_size, channels, num_frames, frequency_size = x.shape
        x = x.transpose(1, 2).reshape(batch_size, num_frames, channels * frequency_size)
        x = self.output_projection(x)

        output_lengths = self._subsample_lengths(feature_lengths)
        frame_indices = torch.arange(num_frames, device=x.device)
        valid_frames = frame_indices[None, :] < output_lengths[:, None]
        x = x.masked_fill(~valid_frames[:, :, None], 0.0)
        return x, output_lengths

    @classmethod
    def _pad_time(cls, x: torch.Tensor) -> torch.Tensor:
        """Left-pad the time axis for a causal kernel-size-three convolution."""
        return F.pad(x, (0, 0, cls._KERNEL_SIZE - 1, 0))  # (batch, channels, 2 + num_frames, input_size)

    @classmethod
    def _subsample_lengths(cls, lengths: torch.Tensor) -> torch.Tensor:
        """Apply three rounds of ``ceil(length / 2)``."""
        for _ in range(cls._NUM_STAGES):
            lengths = torch.div(lengths + cls._STRIDE - 1, cls._STRIDE, rounding_mode="floor")
        return lengths

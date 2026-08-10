from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from asr.modules.conformer.cache import FastConformerSubsamplingCache, SubsamplingStageCache


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

    def forward_chunk(
        self,
        features: torch.Tensor,
        cache: FastConformerSubsamplingCache | None = None,
    ) -> tuple[torch.Tensor, FastConformerSubsamplingCache]:
        """
        Args:
            features: Next unpadded acoustic-feature chunk with shape
                ``(batch, num_frames, input_size)``. Every frame is treated as valid.
            cache: States returned by the preceding call, or ``None`` at the
                beginning of an utterance.

        Returns:
            Newly available subsampled frames and updated states for all three
            convolution stages. Concatenated outputs are equivalent to
            :meth:`forward` on the concatenated input.
        """
        if features.ndim != 3:
            raise ValueError(f"features must have shape (batch, num_frames, input_size), but got {features.shape}")
        if features.shape[-1] != self.input_size:
            raise ValueError(f"expected input_size {self.input_size}, but got {features.shape[-1]}")

        stage_caches: tuple[SubsamplingStageCache | None, ...]
        if cache is None:
            stage_caches = (None,) * self._NUM_STAGES
        else:
            if len(cache.stages) != self._NUM_STAGES:
                raise ValueError(f"cache must contain {self._NUM_STAGES} subsampling stages")
            stage_caches = cache.stages

        x = features.unsqueeze(1)
        x, input_cache = self._forward_stage(x, self.input_conv, stage_caches[0])
        x = self.activation(x)

        next_stage_caches = [input_cache]
        for depthwise_conv, pointwise_conv, stage_cache in zip(
            self.depthwise_convs,
            self.pointwise_convs,
            stage_caches[1:],
            strict=True,
        ):
            x, next_stage_cache = self._forward_stage(x, cast(nn.Conv2d, depthwise_conv), stage_cache)
            if x.shape[2] > 0:
                x = self.activation(pointwise_conv(x))
            next_stage_caches.append(next_stage_cache)

        batch_size, channels, num_frames, frequency_size = x.shape
        x = x.transpose(1, 2).reshape(batch_size, num_frames, channels * frequency_size)
        x = self.output_projection(x)
        return x, FastConformerSubsamplingCache(stages=tuple(next_stage_caches))

    def _forward_stage(
        self,
        x: torch.Tensor,
        convolution: nn.Conv2d,
        cache: SubsamplingStageCache | None,
    ) -> tuple[torch.Tensor, SubsamplingStageCache]:
        expected_context_shape = (
            x.shape[0],
            convolution.in_channels,
            self._KERNEL_SIZE - 1,
            x.shape[3],
        )
        if cache is None:
            context = x.new_zeros(expected_context_shape)
            num_previous_frames = 0
        else:
            if cache.context.shape != expected_context_shape:
                raise ValueError(
                    f"subsampling context must have shape {expected_context_shape}, but got {cache.context.shape}"
                )
            if cache.context.device != x.device or cache.context.dtype != x.dtype:
                raise ValueError("subsampling input and context must have the same dtype and device")
            if cache.num_frames < 0:
                raise ValueError("cached subsampling frame count must be non-negative")
            context = cache.context
            num_previous_frames = cache.num_frames

        convolution_input = torch.cat((context, x), dim=2)
        next_cache = SubsamplingStageCache(
            context=convolution_input[:, :, -(self._KERNEL_SIZE - 1) :],
            num_frames=num_previous_frames + x.shape[2],
        )

        # The global stride grid starts at input frame zero. An odd number of
        # previously received frames shifts the first window by one local frame.
        stride_offset = num_previous_frames % self._STRIDE
        convolution_input = convolution_input[:, :, stride_offset:]
        if convolution_input.shape[2] < self._KERNEL_SIZE:
            output_frequency = (x.shape[3] + self._STRIDE - 1) // self._STRIDE
            output = x.new_empty((x.shape[0], convolution.out_channels, 0, output_frequency))
        else:
            output = convolution(convolution_input)
        return output, next_cache

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

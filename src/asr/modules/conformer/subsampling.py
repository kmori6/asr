from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from asr.modules.conformer.cache import FastConformerSubsamplingCache


class FastConformerSubsampling(nn.Module):
    """Apply non-causal 8x convolutional subsampling to acoustic features.

    Proposed in D. Rekesh et al., "Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition,"
    in Proc. ASRU, 2023.

    """

    _NUM_STAGES = 3
    _KERNEL_SIZE = 3
    _STRIDE = 2
    _CONV_PADDING = (1, 1)

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
            padding=self._CONV_PADDING,
        )
        self.depthwise_convs = nn.ModuleList(
            nn.Conv2d(
                in_channels=conv_channels,
                out_channels=conv_channels,
                kernel_size=self._KERNEL_SIZE,
                stride=self._STRIDE,
                padding=self._CONV_PADDING,
                groups=conv_channels,
            )
            for _ in range(self._NUM_STAGES - 1)
        )
        self.pointwise_convs = nn.ModuleList(
            nn.Conv2d(conv_channels, conv_channels, kernel_size=1) for _ in range(self._NUM_STAGES - 1)
        )
        self.activation = nn.ReLU()
        self.output_projection = nn.Linear(conv_channels * frequency_size, output_size)

    def forward(self, features: torch.Tensor, feature_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            features (torch.Tensor): Acoustic features with shape ``(batch, num_frames, input_size)``.
            feature_lengths (torch.Tensor): Valid frame counts with shape ``(batch,)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Encoded features
                with shape ``(batch, ceil(num_frames / 8), output_size)``
                and valid lengths ``ceil(feature_lengths / 8)``. Invalid output frames are zero.
        """
        self._validate_inputs(features, feature_lengths)

        stage_lengths = feature_lengths
        x = features.unsqueeze(1)  # (batch, 1, num_frames, input_size)
        x = self.activation(self.input_conv(self._prepare_stage_input(x, stage_lengths)))
        stage_lengths = self._subsample_once(stage_lengths)
        for depthwise_conv, pointwise_conv in zip(self.depthwise_convs, self.pointwise_convs, strict=True):
            x = self.activation(pointwise_conv(depthwise_conv(self._prepare_stage_input(x, stage_lengths))))
            stage_lengths = self._subsample_once(stage_lengths)

        x = self._project(x)
        x = self._mask_invalid_outputs(x, stage_lengths)
        return x, stage_lengths

    def _validate_features(self, features: torch.Tensor) -> None:
        if features.ndim != 3:
            raise ValueError(f"features must have shape (batch, num_frames, input_size), but got {features.shape}")
        if features.shape[-1] != self.input_size:
            raise ValueError(f"expected input_size {self.input_size}, but got {features.shape[-1]}")

    def _validate_inputs(self, features: torch.Tensor, feature_lengths: torch.Tensor) -> None:
        self._validate_features(features)
        if feature_lengths.ndim != 1 or feature_lengths.shape[0] != features.shape[0]:
            raise ValueError(
                "feature_lengths must have shape (batch,), "
                f"but got {tuple(feature_lengths.shape)} for batch size {features.shape[0]}"
            )
        if feature_lengths.device != features.device:
            raise ValueError("features and feature_lengths must be on the same device")
        if torch.any(feature_lengths < 0) or torch.any(feature_lengths > features.shape[1]):
            raise ValueError(f"feature lengths must be between 0 and the padded feature length ({features.shape[1]})")

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, num_frames, frequency_size = x.shape
        x = x.transpose(1, 2).reshape(batch_size, num_frames, channels * frequency_size)
        return self.output_projection(x)

    def _prepare_stage_input(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Mask padding before a non-causal convolution can mix it into valid frames."""
        return self._mask_invalid_frames(x, lengths)

    @staticmethod
    def _mask_invalid_frames(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        frame_indices = torch.arange(x.shape[2], device=x.device)
        valid_frames = frame_indices[None, None, :, None] < lengths[:, None, None, None]
        return x.masked_fill(~valid_frames, 0.0)

    @staticmethod
    def _mask_invalid_outputs(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        frame_indices = torch.arange(x.shape[1], device=x.device)
        valid_frames = frame_indices[None, :] < lengths[:, None]
        return x.masked_fill(~valid_frames[:, :, None], 0.0)

    @classmethod
    def _subsample_once(cls, lengths: torch.Tensor) -> torch.Tensor:
        """Apply one round of ``ceil(length / 2)``."""
        return torch.div(lengths + cls._STRIDE - 1, cls._STRIDE, rounding_mode="floor")


class CausalFastConformerSubsampling(FastConformerSubsampling):
    """Apply a causal variant of Fast Conformer's 8x subsampling."""

    _CONV_PADDING = (0, 1)

    def _prepare_stage_input(self, x: torch.Tensor, _lengths: torch.Tensor) -> torch.Tensor:
        return F.pad(x, (0, 0, self._KERNEL_SIZE - 1, 0))

    def forward_chunk(
        self, features: torch.Tensor, cache: FastConformerSubsamplingCache | None = None
    ) -> tuple[torch.Tensor, FastConformerSubsamplingCache | None]:
        """

        Args:
            features (torch.Tensor): Next unpadded acoustic-feature chunk for one utterance
                with shape ``(1, num_frames, input_size)``. Every frame is valid.
            cache (FastConformerSubsamplingCache | None, optional): States returned by the preceding call,
                or ``None`` at the beginning of an utterance.

        Returns:
            tuple[torch.Tensor, FastConformerSubsamplingCache | None]: Newly available
                subsampled frames and updated stage states. The cache remains ``None``
                until at least one input frame arrives. Concatenated outputs are equivalent
                to ``forward`` on the concatenated input.
        """
        self._validate_features(features)
        if features.shape[0] != 1:
            raise ValueError(f"streaming features must have batch size 1, but got {features.shape[0]}")
        if features.shape[1] == 0:
            projection_input = features.new_empty((1, 0, self.output_projection.in_features))
            return self.output_projection(projection_input), cache

        buffers: tuple[torch.Tensor | None, ...]
        if cache is None:
            buffers = (None,) * self._NUM_STAGES
        else:
            if len(cache.buffers) != self._NUM_STAGES:
                raise ValueError(f"cache must contain {self._NUM_STAGES} subsampling buffers")
            buffers = cache.buffers

        x = features.unsqueeze(1)
        x, input_buffer = self._forward_chunk_stage(x, self.input_conv, None, buffers[0])

        next_buffers = [input_buffer]
        for depthwise_conv, pointwise_conv, buffer in zip(
            self.depthwise_convs,
            self.pointwise_convs,
            buffers[1:],
            strict=True,
        ):
            x, next_buffer = self._forward_chunk_stage(
                x,
                cast(nn.Conv2d, depthwise_conv),
                cast(nn.Conv2d, pointwise_conv),
                buffer,
            )
            next_buffers.append(next_buffer)

        x = self._project(x)
        return x, FastConformerSubsamplingCache(buffers=tuple(next_buffers))

    def _forward_chunk_stage(
        self,
        x: torch.Tensor,
        convolution: nn.Conv2d,
        pointwise_convolution: nn.Conv2d | None,
        buffer: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if buffer is None:
            buffer = x.new_zeros((x.shape[0], convolution.in_channels, self._KERNEL_SIZE - 1, x.shape[3]))
        else:
            valid_shape = (
                buffer.ndim == 4
                and buffer.shape[:2] == (x.shape[0], convolution.in_channels)
                and buffer.shape[2] in (1, self._KERNEL_SIZE - 1)
                and buffer.shape[3] == x.shape[3]
            )
            if not valid_shape:
                raise ValueError(
                    "subsampling buffer must have shape "
                    f"({x.shape[0]}, {convolution.in_channels}, 1 or {self._KERNEL_SIZE - 1}, {x.shape[3]})"
                )
            if buffer.device != x.device:
                raise ValueError("subsampling input and buffer must be on the same device")
            if x.shape[2] == 0:
                output_frequency = (x.shape[3] + self._STRIDE - 1) // self._STRIDE
                output = x.new_empty((1, convolution.out_channels, 0, output_frequency))
                return output, buffer
            if buffer.dtype != x.dtype:
                raise ValueError("subsampling input and buffer must have the same dtype")

        convolution_input = torch.cat((buffer, x), dim=2)
        if convolution_input.shape[2] < self._KERNEL_SIZE:
            output_frequency = (x.shape[3] + self._STRIDE - 1) // self._STRIDE
            output = x.new_empty((x.shape[0], convolution.out_channels, 0, output_frequency))
        else:
            output = convolution(convolution_input)

        num_consumed_frames = output.shape[2] * self._STRIDE
        next_buffer = convolution_input[:, :, num_consumed_frames:].clone()
        if pointwise_convolution is not None and output.shape[2] > 0:
            output = pointwise_convolution(output)
        return self.activation(output), next_buffer

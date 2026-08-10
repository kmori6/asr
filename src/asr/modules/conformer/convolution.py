import torch
import torch.nn as nn
import torch.nn.functional as F


class Convolution(nn.Module):
    """Causal convolution module used in a streaming Conformer block."""

    def __init__(self, input_size: int, kernel_size: int, bias: bool = True) -> None:
        super().__init__()
        if input_size <= 0:
            raise ValueError("input_size must be positive")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")

        self.input_size = input_size
        self.cache_size = kernel_size - 1
        self.pointwise_conv1 = nn.Conv1d(input_size, 2 * input_size, kernel_size=1, bias=bias)
        self.glu_activation = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            input_size,
            input_size,
            kernel_size=kernel_size,
            groups=input_size,
            bias=bias,
        )
        self.layernorm = nn.LayerNorm(input_size)
        self.swish_activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(input_size, input_size, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """

        Args:
            x: Input tensor with shape ``(batch, num_frames, input_size)``.
            mask: Boolean tensor with shape ``(batch, num_frames)`` where ``True``
                marks a valid frame.

        Returns:
            Tensor with the same shape as ``x``. Invalid output frames are zero.
        """
        output, _ = self._forward(x, mask=mask, cache=None)
        return output

    def forward_chunk(
        self,
        x: torch.Tensor,
        cache: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply convolution incrementally to a padded chunk.

        Args:
            x: Current input chunk with shape ``(batch, chunk_size, input_size)``.
            cache: Previous post-GLU activations with shape
                ``(batch, input_size, kernel_size - 1)``. ``None`` starts a new
                stream with zero left context.
            mask: Boolean tensor with shape ``(batch, chunk_size)`` where ``True``
                marks a valid frame.

        Returns:
            Current outputs with the same shape as ``x`` and the next fixed-size
            cache. Concatenated chunk outputs are equivalent to a complete-sequence
            forward pass.
        """
        return self._forward(x, mask=mask, cache=cache)

    def _forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
        cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"x must have shape (batch, num_frames, input_size), but got {tuple(x.shape)}")
        if x.shape[-1] != self.input_size:
            raise ValueError(f"expected input_size {self.input_size}, but got {x.shape[-1]}")
        if x.shape[1] == 0:
            raise ValueError("x must contain at least one frame")
        if mask is not None:
            if mask.shape != x.shape[:2] or mask.dtype != torch.bool:
                raise ValueError("mask must be a boolean tensor with shape (batch, num_frames)")
            if mask.device != x.device:
                raise ValueError("x and mask must be on the same device")

        x = self.glu_activation(self.pointwise_conv1(x.transpose(1, 2)))
        if mask is not None:
            x = x.masked_fill(~mask[:, None, :], 0.0)

        expected_cache_shape = (x.shape[0], self.input_size, self.cache_size)
        if cache is None:
            depthwise_input = F.pad(x, (self.cache_size, 0))
        else:
            if cache.shape != expected_cache_shape:
                raise ValueError(f"cache must have shape {expected_cache_shape}, but got {tuple(cache.shape)}")
            if cache.device != x.device or cache.dtype != x.dtype:
                raise ValueError("x and cache must have the same dtype and device")
            depthwise_input = torch.cat((cache, x), dim=-1)

        if self.cache_size == 0:
            next_cache = x.new_empty(expected_cache_shape)
        else:
            next_cache = depthwise_input[:, :, -self.cache_size :]

        x = self.depthwise_conv(depthwise_input).transpose(1, 2)
        x = self.swish_activation(self.layernorm(x))
        x = self.pointwise_conv2(x.transpose(1, 2)).transpose(1, 2)
        if mask is not None:
            x = x.masked_fill(~mask[:, :, None], 0.0)
        return x, next_cache

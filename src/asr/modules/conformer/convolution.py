import torch
import torch.nn as nn
import torch.nn.functional as F


class Convolution(nn.Module):
    """Non-causal convolution module from the original Conformer architecture.

    Proposed in A. Gulati et al., "Conformer: Convolution-augmented Transformer for Speech Recognition,"
    in Proc. Interspeech, 2020, pp. 5036-5040.

    """

    _norm_type: type[nn.Module] = nn.BatchNorm1d

    def __init__(self, input_size: int, kernel_size: int, bias: bool = True) -> None:
        super().__init__()
        if input_size <= 0:
            raise ValueError("input_size must be positive")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        self.input_size = input_size
        self.kernel_size = kernel_size
        self.pointwise_conv1 = nn.Conv1d(input_size, 2 * input_size, kernel_size=1, bias=bias)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            input_size,
            input_size,
            kernel_size=kernel_size,
            padding=self._depthwise_padding(kernel_size),
            groups=input_size,
            bias=bias,
        )
        self.norm = self._norm_type(input_size)
        self.silu = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(input_size, input_size, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Input tensor with shape ``(batch, num_frames, input_size)``.
            mask (torch.Tensor): Boolean tensor with shape
                ``(batch, num_frames)`` where ``True`` marks a valid frame.

        Returns:
            torch.Tensor: Output tensor with the same shape as ``x``. Invalid frames are zero.
        """
        self._validate_inputs(x, mask)
        x = self._input_projection(x, mask)
        x = self._convolve(x)
        return self._mask_invalid_outputs(x, mask)

    @staticmethod
    def _depthwise_padding(kernel_size: int) -> int:
        return (kernel_size - 1) // 2

    def _validate_inputs(self, x: torch.Tensor, mask: torch.Tensor) -> None:
        if x.ndim != 3:
            raise ValueError(f"x must have shape (batch, num_frames, input_size), but got {tuple(x.shape)}")
        if x.shape[-1] != self.input_size:
            raise ValueError(f"expected input_size {self.input_size}, but got {x.shape[-1]}")
        if x.shape[1] == 0:
            raise ValueError("x must contain at least one frame")
        if mask.shape != x.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("mask must be a boolean tensor with shape (batch, num_frames)")
        if mask.device != x.device:
            raise ValueError("x and mask must be on the same device")

    def _input_projection(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.glu(self.pointwise_conv1(x.transpose(1, 2)))
        return x.masked_fill(~mask[:, None, :], 0.0)

    def _convolve(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise_conv(x)
        x = self.silu(self._normalize(x))
        return self.pointwise_conv2(x).transpose(1, 2)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)

    @staticmethod
    def _mask_invalid_outputs(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return x.masked_fill(~mask[:, :, None], 0.0)


class CausalConvolution(Convolution):
    """Causal Conformer convolution with a fixed-size cache for streaming."""

    _norm_type = nn.LayerNorm

    @staticmethod
    def _depthwise_padding(_: int) -> int:
        return 0

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Input tensor with shape ``(batch, num_frames, input_size)``.
            mask (torch.Tensor): Boolean tensor with shape
                ``(batch, num_frames)`` where ``True`` marks a valid frame.

        Returns:
            torch.Tensor: Output tensor with the same shape as ``x``. Invalid frames are zero.
        """
        output, _ = self.forward_chunk(x, mask)
        return output

    def forward_chunk(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            x (torch.Tensor): Current input chunk with shape ``(batch, chunk_size, input_size)``.
            mask (torch.Tensor): Boolean tensor with shape ``(batch, chunk_size)``
                where ``True`` marks a valid frame.
            cache (torch.Tensor | None, optional): Previous post-GLU activations with shape
                ``(batch, input_size, kernel_size - 1)``. ``None`` starts a new
                stream with zero left context.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Current chunk outputs with the same
                shape as ``x`` and the next fixed-size cache. Concatenated chunk
                outputs are equivalent to a complete-sequence forward pass.
        """
        self._validate_inputs(x, mask)
        x = self._input_projection(x, mask)

        cache_size = self.kernel_size - 1
        expected_cache_shape = (x.shape[0], self.input_size, cache_size)
        if cache is None:
            depthwise_input = F.pad(x, (cache_size, 0))
        else:
            if cache.shape != expected_cache_shape:
                raise ValueError(f"cache must have shape {expected_cache_shape}, but got {tuple(cache.shape)}")
            if cache.device != x.device or cache.dtype != x.dtype:
                raise ValueError("x and cache must have the same dtype and device")
            depthwise_input = torch.cat((cache, x), dim=-1)

        cache_start = depthwise_input.shape[-1] - cache_size
        next_cache = depthwise_input[:, :, cache_start:]

        x = self._convolve(depthwise_input)
        return self._mask_invalid_outputs(x, mask), next_cache

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)

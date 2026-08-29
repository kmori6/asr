import math

import torch
import torch.nn as nn

from asr.modules.conformer.cache import KVCache


class MultiHeadSelfAttention(nn.Module):
    """Multi-head attention with relative positional encoding.

    Proposed in Z. Dai et al., "Transformer-XL: attentive language models beyond a fixed-length context,"
    in ACL, 2019, pp. 2978-2988.

    """

    inverse_frequencies: torch.Tensor

    def __init__(self, input_size: int, num_heads: int, dropout_rate: float, bias: bool = True) -> None:
        super().__init__()
        if input_size <= 0:
            raise ValueError("input_size must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if input_size % num_heads != 0:
            raise ValueError("input_size must be divisible by num_heads")
        if input_size % 2 != 0:
            raise ValueError("input_size must be even for sinusoidal positional encoding")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate must satisfy 0 <= dropout_rate < 1")
        self.input_size = input_size
        self.h = num_heads
        self.d_k = input_size // num_heads
        self.w_q = nn.Linear(input_size, input_size, bias=bias)
        self.w_k = nn.Linear(input_size, input_size, bias=bias)
        self.w_v = nn.Linear(input_size, input_size, bias=bias)
        self.w_p = nn.Linear(input_size, input_size, bias=False)
        self.w_o = nn.Linear(input_size, input_size, bias=bias)
        self.b_u = nn.Parameter(torch.zeros(num_heads, self.d_k))
        self.b_v = nn.Parameter(torch.zeros(num_heads, self.d_k))
        self.dropout = nn.Dropout(dropout_rate)

        inverse_frequencies = torch.exp(
            torch.arange(0, input_size, 2, dtype=torch.float32) * (-math.log(10_000.0) / input_size)
        )
        self.register_buffer("inverse_frequencies", inverse_frequencies, persistent=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Input tensor with shape ``(batch, num_frames, input_size)``.
            mask (torch.Tensor): Boolean tensor with shape ``(batch, num_frames, num_frames)``
                where ``True`` marks an allowed query-key pair.

        Returns:
            torch.Tensor: Output tensor with the same shape as ``x``. A fully masked query
                produces an all-zero output.
        """
        output, _ = self.forward_chunk(x, mask)
        return output

    def forward_chunk(
        self, x: torch.Tensor, mask: torch.Tensor, cache: KVCache | None = None
    ) -> tuple[torch.Tensor, KVCache]:
        """

        Args:
            x (torch.Tensor): Current input chunk with shape ``(batch, chunk_size, input_size)``.
            mask (torch.Tensor): Boolean tensor with shape
                ``(batch, chunk_size, cached_length + chunk_size)`` where ``True``
                marks an allowed query-key pair.
            cache (KVCache | None, optional): Projected keys and values from preceding chunks.

        Returns:
            tuple[torch.Tensor, KVCache]: Current chunk outputs and the updated cache
                containing all frames seen so far.
        """
        if x.ndim != 3:
            raise ValueError(f"x must have shape (batch, num_frames, input_size), but got {tuple(x.shape)}")
        if x.shape[-1] != self.input_size:
            raise ValueError(f"expected input_size {self.input_size}, but got {x.shape[-1]}")
        if x.shape[1] == 0:
            raise ValueError("x must contain at least one frame")

        batch_size, query_length = x.shape[:2]
        cache_length = 0
        if cache is not None:
            expected_prefix = (batch_size, self.h)
            expected_suffix = (self.d_k,)
            if (
                cache.key.ndim != 4
                or cache.value.shape != cache.key.shape
                or cache.key.shape[:2] != expected_prefix
                or cache.key.shape[-1:] != expected_suffix
            ):
                raise ValueError("cached keys and values must have shape (batch, num_heads, cached_length, head_size)")
            cache_length = cache.key.shape[2]

        key_length = cache_length + query_length
        expected_mask_shape = (batch_size, query_length, key_length)
        if mask.shape != expected_mask_shape or mask.dtype != torch.bool:
            raise ValueError(f"mask must be a boolean tensor with shape {expected_mask_shape}")
        if mask.device != x.device:
            raise ValueError("x and mask must be on the same device")

        q = self._split_heads(self.w_q(x))
        k = self._split_heads(self.w_k(x))
        v = self._split_heads(self.w_v(x))
        if cache is not None:
            if (
                cache.key.device != k.device
                or cache.value.device != v.device
                or cache.key.dtype != k.dtype
                or cache.value.dtype != v.dtype
            ):
                raise ValueError("current and cached keys and values must have the same dtype and device")
            k = torch.cat((cache.key, k), dim=2)
            v = torch.cat((cache.value, v), dim=2)

        next_cache = KVCache(key=k, value=v)
        positional_encoding = self._relative_positional_encoding(query_length, key_length, reference=q)
        p = self.w_p(positional_encoding).view(-1, self.h, self.d_k).transpose(0, 1)

        content_scores = torch.matmul(q + self.b_u[None, :, None, :], k.transpose(-2, -1))
        position_scores = torch.matmul(q + self.b_v[None, :, None, :], p.transpose(-2, -1))
        position_scores = self._relative_shift(position_scores, key_length)
        scores = (content_scores + position_scores) / math.sqrt(self.d_k)

        attention_mask = mask[:, None, :, :]
        scores = scores.masked_fill(~attention_mask, float("-inf"))
        attention = torch.softmax(scores, dim=-1).masked_fill(~attention_mask, 0.0)
        attention = self.dropout(attention)

        output = torch.matmul(attention, v).transpose(1, 2).reshape(batch_size, query_length, self.input_size)
        output = self.w_o(output)
        valid_queries = mask.any(dim=-1)
        output = output.masked_fill(~valid_queries[:, :, None], 0.0)
        return output, next_cache

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split the model dimension into attention heads."""
        batch_size, num_frames = x.shape[:2]
        return x.view(batch_size, num_frames, self.h, self.d_k).transpose(1, 2)

    def _relative_positional_encoding(
        self, query_length: int, key_length: int, reference: torch.Tensor
    ) -> torch.Tensor:
        """Create sinusoidal encodings for all query-key relative distances."""
        positions = torch.arange(key_length - 1, -query_length, -1, dtype=torch.float32, device=reference.device)
        angles = positions[:, None] * self.inverse_frequencies[None, :]
        return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1).to(dtype=reference.dtype)

    @staticmethod
    def _relative_shift(x: torch.Tensor, key_length: int) -> torch.Tensor:
        """Align relative-position scores with their query-key pairs.

        Example:
            >>> distances = torch.tensor([4, 3, 2, 1, 0, -1, -2])
            >>> x = distances.repeat(3, 1).reshape(1, 1, 3, 7)
            >>> MultiHeadSelfAttention._relative_shift(x, key_length=5)
            tensor([[[[ 2,  1,  0, -1, -2],
                      [ 3,  2,  1,  0, -1],
                      [ 4,  3,  2,  1,  0]]]])

        """
        batch_size, num_heads, query_length, position_length = x.shape
        x = nn.functional.pad(x, (1, 0))
        x = x.view(batch_size, num_heads, position_length + 1, query_length)
        x = x[:, :, 1:].view(batch_size, num_heads, query_length, position_length)
        return x[:, :, :, :key_length]

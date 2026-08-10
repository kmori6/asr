from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class KVCache:
    """Projected key and value tensors cached by a self-attention module."""

    key: torch.Tensor  # (batch, num_heads, num_frames, head_size)
    value: torch.Tensor  # (batch, num_heads, num_frames, head_size)


@dataclass(frozen=True)
class ConformerBlockCache:
    """Attention and convolution states cached by one Conformer block."""

    attention: KVCache
    convolution: torch.Tensor  # (batch, input_size, kernel_size - 1)

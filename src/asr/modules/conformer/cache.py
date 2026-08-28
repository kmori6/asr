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


@dataclass(frozen=True)
class FastConformerSubsamplingCache:
    """Unconsumed convolution inputs cached by the subsampling stages."""

    buffers: tuple[torch.Tensor, ...]  # Each is (batch, input_channels, 1 or 2, input_frequency).


@dataclass(frozen=True)
class FastConformerCache:
    """Subsampling, pending-chunk, and block states cached by the encoder."""

    subsampling: FastConformerSubsamplingCache | None
    pending: torch.Tensor  # (1, fewer_than_chunk_size, hidden_size)
    blocks: tuple[ConformerBlockCache, ...]
    chunk_size: int

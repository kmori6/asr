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
class SubsamplingStageCache:
    """Left context and stride phase cached by one subsampling stage."""

    context: torch.Tensor  # (batch, input_channels, kernel_size - 1, input_frequency)
    num_frames: int


@dataclass(frozen=True)
class FastConformerSubsamplingCache:
    """States cached by the three FastConformer subsampling stages."""

    stages: tuple[SubsamplingStageCache, ...]


@dataclass(frozen=True)
class FastConformerEncoderCache:
    """Subsampling, pending-chunk, and block states cached by the encoder."""

    subsampling: FastConformerSubsamplingCache | None
    pending: torch.Tensor  # (1, fewer_than_chunk_size, hidden_size)
    blocks: tuple[ConformerBlockCache, ...]
    chunk_size: int

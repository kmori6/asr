from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class KVCache:
    """Projected key and value tensors cached by an attention module."""

    key: torch.Tensor  # (batch_size, num_heads, sequence_length, head_dim)
    value: torch.Tensor  # (batch_size, num_heads, sequence_length, head_dim)


@dataclass(frozen=True)
class DecoderLayerCache:
    """Self- and cross-attention caches for one decoder layer."""

    self_attention: KVCache
    cross_attention: KVCache

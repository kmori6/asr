from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class KVCache:
    """Projected key and value tensors cached by a self-attention module."""

    key: torch.Tensor  # (batch, num_heads, num_frames, head_size)
    value: torch.Tensor  # (batch, num_heads, num_frames, head_size)

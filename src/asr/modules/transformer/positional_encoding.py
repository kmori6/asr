import math

import torch
import torch.nn as nn


def sinusoidal_positional_encoding(
    d_model: int,
    max_length: int,
    base: float = 10000.0,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Sinusoidal positional encoding.

    PE_{(pos, 2i)} = sin(pos/10000^{2i/d_model})
    PE_{(pos, 2i + 1)} = cos(pos/10000^{2i/d_model})

    Args:
        d_model (int): Hidden state dimension.
        max_length (int): Maximum sequence length.
        base (float): Base value for frequency calculation.
        device (torch.device | str | None): Device on which to create the encoding.

    Returns:
        torch.Tensor: Sinusoidal positional encoding (max_length, d_model).
    """
    if d_model <= 0:
        raise ValueError(f"d_model must be positive, but got {d_model}.")
    if max_length < 0:
        raise ValueError(f"max_length must be non-negative, but got {max_length}.")
    if not math.isfinite(base) or base <= 0:
        raise ValueError(f"base must be finite and positive, but got {base}.")

    positions = torch.arange(max_length, dtype=torch.float32, device=device)[:, None]
    dimension_indices = torch.arange(0, d_model, 2, dtype=torch.float32, device=device)
    inverse_frequencies = torch.exp(-math.log(base) * dimension_indices / d_model)
    angles = positions * inverse_frequencies

    encoding = torch.empty(max_length, d_model, dtype=torch.float32, device=device)
    encoding[:, 0::2] = torch.sin(angles)
    encoding[:, 1::2] = torch.cos(angles[:, : d_model // 2])
    return encoding


class PositionalEncoding(nn.Module):
    """Positional encoding module.

    Proposed in A. Vaswani et al., "Attention is all you need," in NeurIPS, 2017, pp. 5998-6008.

    """

    pe: torch.Tensor

    def __init__(self, hidden_size: int, max_length: int = 4096, base: float = 10000.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.base = base
        self.register_buffer("pe", sinusoidal_positional_encoding(hidden_size, max_length, base), persistent=False)

    @property
    def max_length(self) -> int:
        """Current number of precomputed positions."""
        return self.pe.shape[0]

    def _ensure_capacity(self, required_length: int) -> None:
        if required_length <= self.max_length:
            return

        new_length = max(required_length, max(1, self.max_length * 2))
        encoding = sinusoidal_positional_encoding(
            self.hidden_size,
            new_length,
            self.base,
            device=self.pe.device,
        )
        self.pe = encoding.to(dtype=self.pe.dtype)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Embedding tensor (*, sequence_length, hidden_size).
            offset (int): Position of the first token in ``x``.

        Returns:
            torch.Tensor: Positional encoding (1, sequence_length, hidden_size).
        """
        if x.ndim < 2:
            raise ValueError(f"x must have at least 2 dimensions, but got shape {tuple(x.shape)}.")
        if x.shape[-1] != self.hidden_size:
            raise ValueError(f"x.shape[-1] must equal hidden_size ({self.hidden_size}), but got {x.shape[-1]}.")
        if offset < 0:
            raise ValueError(f"offset must be non-negative, but got {offset}.")

        end = offset + x.shape[-2]
        self._ensure_capacity(end)
        return self.pe[None, offset:end, :].to(dtype=x.dtype)

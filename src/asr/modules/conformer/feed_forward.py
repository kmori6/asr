import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Position-wise feed-forward network used in a Conformer block."""

    def __init__(self, input_size: int, hidden_size: int, dropout_rate: float, bias: bool = True) -> None:
        super().__init__()
        if input_size <= 0:
            raise ValueError("input_size must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate must satisfy 0 <= dropout_rate < 1")

        self.w_1 = nn.Linear(input_size, hidden_size, bias=bias)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.w_2 = nn.Linear(hidden_size, input_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform each frame independently.

        Args:
            x: Input tensor with shape ``(..., input_size)``.

        Returns:
            Tensor with the same shape, dtype, and device as ``x``.
        """
        x = self.w_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.w_2(x)

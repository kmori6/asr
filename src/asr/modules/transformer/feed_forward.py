import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Position-wise feed-forward networks.

    Proposed in A. Vaswani et al., "Attention is all you need," in NeurIPS, 2017, pp. 5998-6008.

    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout_rate: float,
        activation: nn.Module = nn.ReLU(),
        bias: bool = True,
    ):
        super().__init__()
        if input_size <= 0:
            raise ValueError("input_size must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate must satisfy 0 <= dropout_rate < 1")

        self.w_1 = nn.Linear(input_size, hidden_size, bias=bias)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)
        self.w_2 = nn.Linear(hidden_size, input_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Input tensor with shape ``(..., input_size)``.

        Returns:
            torch.Tensor: Tensor with the same shape, dtype, and device as ``x``.
        """
        x = self.w_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.w_2(x)
        return x

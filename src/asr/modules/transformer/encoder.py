import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, layers: list[nn.Module], final_norm: nn.Module) -> None:
        super().__init__()
        if not layers:
            raise ValueError("layers must not be empty.")

        self.layers = nn.ModuleList(layers)
        self.final_norm = final_norm

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Input tensor (batch_size, sequence_length, d_model).
            mask (torch.Tensor): Attention mask (batch_size, sequence_length, sequence_length).

        Returns:
            torch.Tensor: Output tensor (batch_size, sequence_length, d_model).
        """
        for layer in self.layers:
            x = layer(x, mask)

        return self.final_norm(x)

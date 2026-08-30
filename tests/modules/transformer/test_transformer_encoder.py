import torch
import torch.nn as nn

from asr.modules.transformer.encoder import Encoder


class _Scale(nn.Module):
    def __init__(self, factor: float) -> None:
        super().__init__()
        self.factor = factor

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return x * self.factor


class _Shift(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.value


def test_encoder_applies_layers_in_order_and_final_norm() -> None:
    encoder = Encoder(
        layers=[_Scale(2.0), _Scale(3.0)],
        final_norm=_Shift(1.0),
    )
    inputs = torch.ones(2, 3, 4)
    mask = torch.ones(2, 3, 3, dtype=torch.bool)

    outputs = encoder(inputs, mask)

    torch.testing.assert_close(outputs, inputs * 6.0 + 1.0)

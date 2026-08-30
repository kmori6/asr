import torch
import torch.nn as nn

from asr.modules.transformer.encoder_layer import EncoderLayer
from asr.modules.transformer.feed_forward import FeedForward
from asr.modules.transformer.multi_head_attention import MultiHeadAttention


def test_encoder_layer_uses_pre_norm_and_propagates_gradients() -> None:
    module = EncoderLayer(
        mha=MultiHeadAttention(
            hidden_size=4,
            num_heads=2,
            dropout_rate=0.0,
        ),
        mha_norm=nn.LayerNorm(4),
        ffn=FeedForward(input_size=4, hidden_size=16, dropout_rate=0.0),
        ffn_norm=nn.LayerNorm(4),
        dropout_rate=0.0,
    )
    inputs = torch.randn(2, 3, 4, requires_grad=True)
    mask = torch.ones(2, 3, 3, dtype=torch.bool)

    outputs = module(inputs, mask)

    with torch.no_grad():
        normalized = module.mha_norm(inputs)
        expected = inputs + module.mha(normalized, normalized, normalized, mask)
        expected = expected + module.ffn(module.ffn_norm(expected))
    torch.testing.assert_close(outputs, expected)
    assert outputs.shape == inputs.shape

    outputs.sum().backward()
    assert inputs.grad is not None
    assert all(parameter.grad is not None for parameter in module.parameters())

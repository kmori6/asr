import torch
import torch.nn as nn

from asr.modules.transformer.decoder_layer import DecoderLayer
from asr.modules.transformer.feed_forward import FeedForward
from asr.modules.transformer.multi_head_attention import MultiHeadAttention


def _create_decoder_layer() -> DecoderLayer:
    return DecoderLayer(
        self_mha=MultiHeadAttention(4, 2, dropout_rate=0.0),
        self_mha_norm=nn.LayerNorm(4),
        cross_mha=MultiHeadAttention(4, 2, dropout_rate=0.0),
        cross_mha_norm=nn.LayerNorm(4),
        ffn=FeedForward(4, 16, dropout_rate=0.0),
        ffn_norm=nn.LayerNorm(4),
        dropout_rate=0.0,
    )


def test_decoder_layer_uses_pre_norm_and_propagates_gradients() -> None:
    module = _create_decoder_layer()
    encoder_outputs = torch.randn(2, 3, 4, requires_grad=True)
    inputs = torch.randn(2, 4, 4, requires_grad=True)
    encoder_mask = torch.ones(2, 4, 3, dtype=torch.bool)
    decoder_mask = torch.ones(2, 4, 4, dtype=torch.bool).tril()

    outputs = module(encoder_outputs, inputs, encoder_mask, decoder_mask)

    with torch.no_grad():
        normalized = module.self_mha_norm(inputs)
        expected = inputs + module.self_mha(normalized, normalized, normalized, decoder_mask)
        normalized = module.cross_mha_norm(expected)
        expected = expected + module.cross_mha(normalized, encoder_outputs, encoder_outputs, encoder_mask)
        expected = expected + module.ffn(module.ffn_norm(expected))
    torch.testing.assert_close(outputs, expected)

    outputs.sum().backward()
    assert inputs.grad is not None
    assert encoder_outputs.grad is not None
    assert all(parameter.grad is not None for parameter in module.parameters())


def test_decoder_layer_predict_matches_causal_forward() -> None:
    module = _create_decoder_layer().eval()
    encoder_outputs = torch.randn(2, 3, 4)
    inputs = torch.randn(2, 4, 4)
    encoder_mask = torch.ones(2, 4, 3, dtype=torch.bool)
    decoder_mask = torch.ones(2, 4, 4, dtype=torch.bool).tril()

    expected = module(encoder_outputs, inputs, encoder_mask, decoder_mask)
    cache = None
    step_outputs = []
    for position in range(inputs.shape[1]):
        output, cache = module.predict(
            x_enc=encoder_outputs,
            x_dec=inputs[:, position : position + 1],
            mask_enc=encoder_mask[:, position : position + 1],
            mask_dec=decoder_mask[:, position : position + 1, : position + 1],
            cache=cache,
        )
        step_outputs.append(output)

    torch.testing.assert_close(torch.cat(step_outputs, dim=1), expected)
    assert cache is not None
    assert cache.self_attention.key.shape == (2, 2, 4, 2)
    assert cache.self_attention.value.shape == (2, 2, 4, 2)
    assert cache.cross_attention.key.shape == (2, 2, 3, 2)
    assert cache.cross_attention.value.shape == (2, 2, 3, 2)

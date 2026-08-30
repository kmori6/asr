import pytest
import torch
import torch.nn as nn

from asr.modules.transformer.cache import DecoderLayerCache
from asr.modules.transformer.decoder import Decoder
from asr.modules.transformer.decoder_layer import DecoderLayer
from asr.modules.transformer.feed_forward import FeedForward
from asr.modules.transformer.multi_head_attention import MultiHeadAttention


def _create_decoder() -> Decoder:
    return Decoder(
        layers=[
            DecoderLayer(
                self_mha=MultiHeadAttention(4, 2, dropout_rate=0.0),
                self_mha_norm=nn.LayerNorm(4),
                cross_mha=MultiHeadAttention(4, 2, dropout_rate=0.0),
                cross_mha_norm=nn.LayerNorm(4),
                ffn=FeedForward(4, 16, dropout_rate=0.0),
                ffn_norm=nn.LayerNorm(4),
                dropout_rate=0.0,
            )
            for _ in range(2)
        ],
        final_norm=nn.LayerNorm(4),
    )


def test_decoder_predict_matches_causal_forward() -> None:
    decoder = _create_decoder().eval()
    encoder_outputs = torch.randn(2, 3, 4)
    inputs = torch.randn(2, 4, 4)
    attention_mask = torch.tensor([[True, True, False], [True, True, True]])
    encoder_mask = attention_mask[:, None, :].expand(-1, inputs.shape[1], -1)
    decoder_mask = torch.ones(2, 4, 4, dtype=torch.bool).tril()

    expected = decoder(encoder_outputs, inputs, encoder_mask, decoder_mask)
    caches: list[DecoderLayerCache] = []
    step_outputs = []
    cross_cache_pointers = []
    for position in range(inputs.shape[1]):
        output, caches = decoder.predict(
            x_enc=encoder_outputs,
            x_dec=inputs[:, position : position + 1],
            attention_mask=attention_mask,
            caches=caches,
        )
        step_outputs.append(output)
        cross_cache_pointers.append(
            [(cache.cross_attention.key.data_ptr(), cache.cross_attention.value.data_ptr()) for cache in caches]
        )

    torch.testing.assert_close(torch.cat(step_outputs, dim=1), expected)
    assert len(caches) == 2
    assert all(cache.self_attention.key.shape == (2, 2, 4, 2) for cache in caches)
    assert all(cache.cross_attention.key.shape == (2, 2, 3, 2) for cache in caches)
    assert all(pointers == cross_cache_pointers[0] for pointers in cross_cache_pointers)


def test_decoder_predict_requires_one_token() -> None:
    decoder = Decoder(layers=[nn.Identity()], final_norm=nn.Identity())

    with pytest.raises(ValueError, match="exactly one token"):
        decoder.predict(
            x_enc=torch.randn(1, 3, 4),
            x_dec=torch.randn(1, 2, 4),
            attention_mask=torch.ones(1, 3, dtype=torch.bool),
            caches=[],
        )

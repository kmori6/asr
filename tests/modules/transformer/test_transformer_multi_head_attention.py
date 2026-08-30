import math

import pytest
import torch

from asr.modules.transformer.multi_head_attention import MultiHeadAttention


def test_multi_head_attention_validates_configuration() -> None:
    with pytest.raises(ValueError, match="hidden_size must be positive"):
        MultiHeadAttention(hidden_size=0, num_heads=2, dropout_rate=0.0)
    with pytest.raises(ValueError, match="num_heads must be positive"):
        MultiHeadAttention(hidden_size=4, num_heads=0, dropout_rate=0.0)
    with pytest.raises(ValueError, match="hidden_size must be divisible"):
        MultiHeadAttention(hidden_size=5, num_heads=2, dropout_rate=0.0)
    with pytest.raises(ValueError, match="dropout_rate"):
        MultiHeadAttention(hidden_size=4, num_heads=2, dropout_rate=1.0)


def test_multi_head_attention_computation_mask_and_gradient() -> None:
    module = MultiHeadAttention(
        hidden_size=4,
        num_heads=2,
        dropout_rate=0.0,
    )
    query = torch.randn(2, 3, 4, requires_grad=True)
    key = torch.randn(2, 4, 4, requires_grad=True)
    value = torch.randn(2, 4, 4, requires_grad=True)
    mask = torch.ones(2, 3, 4, dtype=torch.bool)
    mask[:, 0, -1] = False

    outputs = module(query, key, value, mask)

    with torch.no_grad():
        projected_query = module._split_heads(module.w_q(query))
        projected_key = module._split_heads(module.w_k(key))
        projected_value = module._split_heads(module.w_v(value))
        scores = projected_query @ projected_key.transpose(-2, -1) / math.sqrt(module.d_k)
        scores = scores.masked_fill(~mask[:, None, :, :], float("-inf"))
        attention = torch.softmax(scores, dim=-1)
        expected = attention @ projected_value
        expected = module.w_o(expected.transpose(1, 2).flatten(2))
    torch.testing.assert_close(outputs, expected)
    assert outputs.shape == query.shape

    outputs.sum().backward()
    assert query.grad is not None
    assert key.grad is not None
    assert value.grad is not None
    assert all(parameter.grad is not None for parameter in module.parameters())


def test_multi_head_attention_kv_cache_matches_causal_attention() -> None:
    module = MultiHeadAttention(
        hidden_size=8,
        num_heads=4,
        dropout_rate=0.0,
    ).eval()
    inputs = torch.randn(2, 4, 8)
    mask = torch.ones(2, 4, 4, dtype=torch.bool).tril()

    expected = module(inputs, inputs, inputs, mask)
    cache = None
    step_outputs = []
    for position in range(inputs.shape[1]):
        output, cache = module.forward_with_cache(
            q=inputs[:, position : position + 1],
            k=inputs[:, position : position + 1],
            v=inputs[:, position : position + 1],
            mask=mask[:, position : position + 1, : position + 1],
            cache=cache,
        )
        step_outputs.append(output)

    torch.testing.assert_close(torch.cat(step_outputs, dim=1), expected)
    assert cache is not None
    projected_key = module._split_heads(module.w_k(inputs))
    projected_value = module._split_heads(module.w_v(inputs))
    torch.testing.assert_close(cache.key, projected_key)
    torch.testing.assert_close(cache.value, projected_value)
    assert cache.key.shape == (2, 4, 4, 2)
    assert cache.value.shape == (2, 4, 4, 2)
    assert cache.key.data_ptr() != cache.value.data_ptr()


def test_multi_head_attention_static_kv_cache_matches_cross_attention() -> None:
    module = MultiHeadAttention(
        hidden_size=8,
        num_heads=4,
        dropout_rate=0.0,
    ).eval()
    query = torch.randn(2, 4, 8)
    key = torch.randn(2, 3, 8)
    value = torch.randn(2, 3, 8)
    mask = torch.ones(2, 4, 3, dtype=torch.bool)

    expected = module(query, key, value, mask)
    cache = None
    step_outputs = []
    cache_pointers = []
    key_projection_outputs = []
    value_projection_outputs = []
    key_hook = module.w_k.register_forward_hook(lambda _module, _inputs, output: key_projection_outputs.append(output))
    value_hook = module.w_v.register_forward_hook(
        lambda _module, _inputs, output: value_projection_outputs.append(output)
    )
    for position in range(query.shape[1]):
        output, cache = module.forward_with_static_cache(
            q=query[:, position : position + 1],
            k=key,
            v=value,
            mask=mask[:, position : position + 1],
            cache=cache,
        )
        step_outputs.append(output)
        cache_pointers.append((cache.key.data_ptr(), cache.value.data_ptr()))
    key_hook.remove()
    value_hook.remove()

    torch.testing.assert_close(torch.cat(step_outputs, dim=1), expected)
    assert cache is not None
    projected_key = module._split_heads(module.w_k(key))
    projected_value = module._split_heads(module.w_v(value))
    torch.testing.assert_close(cache.key, projected_key)
    torch.testing.assert_close(cache.value, projected_value)
    assert len(set(cache_pointers)) == 1
    assert len(key_projection_outputs) == 1
    assert len(value_projection_outputs) == 1

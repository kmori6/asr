import torch

from asr.modules.conformer import MultiHeadSelfAttention


def test_multi_head_self_attention_chunk_outputs_match_full_causal_output() -> None:
    torch.manual_seed(0)
    attention = MultiHeadSelfAttention(input_size=8, num_heads=2, dropout_rate=0.0)
    inputs = torch.randn(2, 7, 8, requires_grad=True)
    lengths = torch.tensor([7, 5])
    indices = torch.arange(inputs.shape[1])
    valid = indices[None, :] < lengths[:, None]
    causal = indices[:, None] >= indices[None, :]
    mask = valid[:, :, None] & valid[:, None, :] & causal[None, :, :]

    full_outputs = attention(inputs, mask)

    cache = None
    chunk_outputs = []
    start = 0
    for chunk in inputs.detach().split((3, 4), dim=1):
        end = start + chunk.shape[1]
        chunk_mask = mask[:, start:end, :end]
        output, cache = attention.forward_chunk(chunk, chunk_mask, cache)
        chunk_outputs.append(output)
        start = end

    assert full_outputs.shape == inputs.shape
    torch.testing.assert_close(torch.cat(chunk_outputs, dim=1), full_outputs.detach())
    torch.testing.assert_close(full_outputs[1, 5:], torch.zeros_like(full_outputs[1, 5:]))
    assert cache is not None
    assert cache.key.shape == (2, 2, 7, 4)
    assert cache.value.shape == cache.key.shape

    full_outputs.sum().backward()
    assert inputs.grad is not None
    assert all(parameter.grad is not None for parameter in attention.parameters())

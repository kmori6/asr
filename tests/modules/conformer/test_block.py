import torch
import torch.nn as nn

from asr.modules.conformer import CausalConvolution, ConformerBlock, Convolution, StreamingConformerBlock


def test_conformer_block_uses_non_causal_convolution() -> None:
    block = ConformerBlock(
        input_size=8,
        num_heads=2,
        kernel_size=5,
        dropout_rate=0.0,
    ).eval()
    inputs = torch.randn(2, 7, 8)
    mask = torch.ones(2, 7, 7, dtype=torch.bool)

    outputs = block(inputs, mask)

    assert type(block.conv) is Convolution
    assert isinstance(block.ffn1.activation, nn.SiLU)
    assert isinstance(block.ffn2.activation, nn.SiLU)
    assert outputs.shape == inputs.shape


def test_streaming_conformer_block_chunk_outputs_match_full_causal_output() -> None:
    torch.manual_seed(0)
    block = StreamingConformerBlock(
        input_size=8,
        num_heads=2,
        kernel_size=5,
        dropout_rate=0.0,
    )
    assert isinstance(block, ConformerBlock)
    assert isinstance(block.conv, CausalConvolution)
    inputs = torch.randn(2, 7, 8, requires_grad=True)
    lengths = torch.tensor([7, 5])
    indices = torch.arange(inputs.shape[1])
    valid = indices[None, :] < lengths[:, None]
    causal = indices[:, None] >= indices[None, :]
    mask = valid[:, :, None] & valid[:, None, :] & causal[None, :, :]

    full_outputs = block(inputs, mask)

    cache = None
    chunk_outputs = []
    start = 0
    for chunk in inputs.detach().split((3, 4), dim=1):
        end = start + chunk.shape[1]
        chunk_mask = mask[:, start:end, :end]
        output, cache = block.forward_chunk(chunk, chunk_mask, cache)
        chunk_outputs.append(output)
        start = end

    assert full_outputs.shape == inputs.shape
    torch.testing.assert_close(torch.cat(chunk_outputs, dim=1), full_outputs.detach())
    torch.testing.assert_close(full_outputs[1, 5:], torch.zeros_like(full_outputs[1, 5:]))
    assert cache is not None
    assert cache.attention.key.shape == (2, 2, 7, 4)
    assert cache.attention.value.shape == cache.attention.key.shape
    assert cache.convolution.shape == (2, 8, 4)

    full_outputs.sum().backward()
    assert inputs.grad is not None
    assert all(parameter.grad is not None for parameter in block.parameters())

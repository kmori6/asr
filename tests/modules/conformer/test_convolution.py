import pytest
import torch
import torch.nn as nn

from asr.modules.conformer import CausalConvolution, Convolution


def test_convolution_applies_non_causal_same_padded_depthwise_convolution() -> None:
    torch.manual_seed(0)
    convolution = Convolution(input_size=4, kernel_size=5).eval()
    inputs = torch.randn(2, 11, 4, requires_grad=True)
    mask = torch.tensor([[True] * 11, [True] * 7 + [False] * 4])

    outputs = convolution(inputs, mask)

    assert outputs.shape == inputs.shape
    assert isinstance(convolution.batchnorm, nn.BatchNorm1d)
    assert convolution.depthwise_conv.padding == (2,)
    torch.testing.assert_close(outputs[1, 7:], torch.zeros_like(outputs[1, 7:]))

    changed_inputs = inputs.detach().clone()
    changed_inputs[:, 6] += 10.0
    changed_outputs = convolution(changed_inputs, mask)
    assert not torch.allclose(outputs[:, 5], changed_outputs[:, 5])

    outputs.sum().backward()
    assert inputs.grad is not None
    assert all(parameter.grad is not None for parameter in convolution.parameters())


def test_convolution_rejects_even_kernel_size() -> None:
    with pytest.raises(ValueError, match="kernel_size must be odd"):
        Convolution(input_size=4, kernel_size=4)


def test_causal_convolution_chunk_outputs_match_full_output() -> None:
    torch.manual_seed(0)
    convolution = CausalConvolution(input_size=4, kernel_size=5)
    inputs = torch.randn(2, 11, 4, requires_grad=True)

    assert isinstance(convolution, Convolution)
    assert isinstance(convolution.layernorm, nn.LayerNorm)
    assert convolution.depthwise_conv.padding == (0,)

    full_outputs = convolution(inputs)

    cache = None
    chunk_outputs = []
    for chunk in inputs.detach().split((3, 4, 4), dim=1):
        output, cache = convolution.forward_chunk(chunk, cache)
        chunk_outputs.append(output)

    assert full_outputs.shape == inputs.shape
    torch.testing.assert_close(torch.cat(chunk_outputs, dim=1), full_outputs.detach())
    assert cache is not None
    assert cache.shape == (2, 4, 4)

    mask = torch.tensor([[True] * 11, [True] * 7 + [False] * 4])
    masked_outputs = convolution(inputs.detach(), mask)
    torch.testing.assert_close(masked_outputs[1, 7:], torch.zeros_like(masked_outputs[1, 7:]))

    full_outputs.sum().backward()
    assert inputs.grad is not None
    assert all(parameter.grad is not None for parameter in convolution.parameters())


def test_causal_convolution_with_kernel_size_one_returns_empty_cache() -> None:
    torch.manual_seed(0)
    convolution = CausalConvolution(input_size=4, kernel_size=1)
    inputs = torch.randn(2, 5, 4)

    full_outputs = convolution(inputs)
    chunk_outputs = []
    cache = None
    for chunk in inputs.split((2, 3), dim=1):
        output, cache = convolution.forward_chunk(chunk, cache)
        chunk_outputs.append(output)

    torch.testing.assert_close(torch.cat(chunk_outputs, dim=1), full_outputs)
    assert cache is not None
    assert cache.shape == (2, 4, 0)

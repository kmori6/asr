import pytest
import torch

from asr.modules.conformer import CausalConvolution, Convolution


def test_convolution_applies_non_causal_same_padded_depthwise_convolution() -> None:
    torch.manual_seed(0)
    convolution = Convolution(input_size=4, kernel_size=5).eval()
    inputs = torch.randn(2, 11, 4, requires_grad=True)
    mask = torch.tensor([[True] * 11, [True] * 7 + [False] * 4])

    outputs = convolution(inputs, mask)

    assert outputs.shape == inputs.shape
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
    mask = torch.ones(2, 11, dtype=torch.bool)

    assert isinstance(convolution, Convolution)

    full_outputs = convolution(inputs, mask)

    cache = None
    chunk_outputs = []
    start = 0
    for chunk in inputs.detach().split((3, 4, 4), dim=1):
        end = start + chunk.shape[1]
        output, cache = convolution.forward_chunk(chunk, mask[:, start:end], cache)
        chunk_outputs.append(output)
        start = end

    assert full_outputs.shape == inputs.shape
    torch.testing.assert_close(torch.cat(chunk_outputs, dim=1), full_outputs.detach())
    assert cache is not None
    assert cache.shape == (2, 4, 4)

    padded_mask = torch.tensor([[True] * 11, [True] * 7 + [False] * 4])
    masked_outputs = convolution(inputs.detach(), padded_mask)
    torch.testing.assert_close(masked_outputs[1, 7:], torch.zeros_like(masked_outputs[1, 7:]))

    full_outputs.sum().backward()
    assert inputs.grad is not None
    assert all(parameter.grad is not None for parameter in convolution.parameters())


def test_causal_convolution_with_kernel_size_one_returns_empty_cache() -> None:
    torch.manual_seed(0)
    convolution = CausalConvolution(input_size=4, kernel_size=1)
    inputs = torch.randn(2, 5, 4)
    mask = torch.ones(2, 5, dtype=torch.bool)

    full_outputs = convolution(inputs, mask)
    chunk_outputs = []
    cache = None
    start = 0
    for chunk in inputs.split((2, 3), dim=1):
        end = start + chunk.shape[1]
        output, cache = convolution.forward_chunk(chunk, mask[:, start:end], cache)
        chunk_outputs.append(output)
        start = end

    torch.testing.assert_close(torch.cat(chunk_outputs, dim=1), full_outputs)
    assert cache is not None
    assert cache.shape == (2, 4, 0)

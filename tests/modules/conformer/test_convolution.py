import torch

from asr.modules.conformer import Convolution


def test_convolution_chunk_outputs_match_full_causal_output() -> None:
    torch.manual_seed(0)
    convolution = Convolution(input_size=4, kernel_size=5)
    inputs = torch.randn(2, 11, 4, requires_grad=True)

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

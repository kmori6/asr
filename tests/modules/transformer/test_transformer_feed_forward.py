import torch

from asr.modules.transformer.feed_forward import FeedForward


def test_feed_forward_computation_and_gradient() -> None:
    module = FeedForward(input_size=4, hidden_size=16, dropout_rate=0.0)
    inputs = torch.randn(2, 3, 4, requires_grad=True)

    outputs = module(inputs)

    with torch.no_grad():
        expected = module.w_2(module.activation(module.w_1(inputs)))
    torch.testing.assert_close(outputs, expected)
    assert outputs.shape == inputs.shape

    outputs.sum().backward()
    assert inputs.grad is not None
    assert all(parameter.grad is not None for parameter in module.parameters())

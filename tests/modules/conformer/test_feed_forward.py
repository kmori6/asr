import torch

from asr.modules.conformer import FeedForward


def test_feed_forward_computation_and_gradient() -> None:
    feed_forward = FeedForward(input_size=4, hidden_size=16, dropout_rate=0.0)
    inputs = torch.randn(2, 3, 4, requires_grad=True)

    outputs = feed_forward(inputs)

    with torch.no_grad():
        expected = feed_forward.w_2(feed_forward.activation(feed_forward.w_1(inputs)))
    torch.testing.assert_close(outputs, expected)
    assert outputs.shape == inputs.shape

    outputs.sum().backward()
    assert inputs.grad is not None
    assert all(parameter.grad is not None for parameter in feed_forward.parameters())

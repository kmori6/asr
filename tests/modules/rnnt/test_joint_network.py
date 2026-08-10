import torch

from asr.modules.rnnt import JointNetwork


def test_joint_network_combines_every_encoder_predictor_pair() -> None:
    torch.manual_seed(0)
    network = JointNetwork(
        vocab_size=16,
        encoder_size=8,
        predictor_size=6,
        hidden_size=10,
        dropout_rate=0.0,
    )
    encoder_outputs = torch.randn(2, 5, 8, requires_grad=True)
    predictor_outputs = torch.randn(2, 4, 6, requires_grad=True)

    logits = network(encoder_outputs, predictor_outputs)
    pair_logits = network(encoder_outputs[:, 2:3], predictor_outputs[:, 1:2])

    assert logits.shape == (2, 5, 4, 16)
    torch.testing.assert_close(pair_logits[:, 0, 0], logits[:, 2, 1])

    with torch.autocast("cpu", dtype=torch.bfloat16):
        mixed_precision_logits = network(encoder_outputs, predictor_outputs.to(torch.bfloat16))
    assert mixed_precision_logits.dtype == torch.bfloat16

    logits.sum().backward()
    assert encoder_outputs.grad is not None
    assert predictor_outputs.grad is not None
    assert all(parameter.grad is not None for parameter in network.parameters())

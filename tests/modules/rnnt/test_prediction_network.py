import torch

from asr.modules.rnnt import PredictionNetwork


def test_prediction_network_incremental_outputs_match_full_outputs() -> None:
    torch.manual_seed(0)
    network = PredictionNetwork(
        vocab_size=16,
        hidden_size=8,
        num_layers=2,
        dropout_rate=0.0,
        blank_token_id=0,
    )
    tokens = torch.tensor([[0, 3, 5, 2], [0, 7, 1, 0]])

    outputs, state = network(tokens)

    incremental_outputs = []
    incremental_state = None
    for token in tokens.split(1, dim=1):
        output, incremental_state = network(token, incremental_state)
        incremental_outputs.append(output)

    assert outputs.shape == (2, 4, 8)
    assert state[0].shape == (2, 2, 8)
    assert state[1].shape == state[0].shape
    torch.testing.assert_close(torch.cat(incremental_outputs, dim=1), outputs)
    assert incremental_state is not None
    torch.testing.assert_close(incremental_state[0], state[0])
    torch.testing.assert_close(incremental_state[1], state[1])

    with torch.autocast("cpu", dtype=torch.bfloat16):
        _, mixed_precision_state = network(tokens[:, :1])
        mixed_precision_output, _ = network(tokens[:, 1:2], mixed_precision_state)
    assert mixed_precision_output.dtype == torch.bfloat16

    outputs.sum().backward()
    assert all(parameter.grad is not None for parameter in network.parameters())

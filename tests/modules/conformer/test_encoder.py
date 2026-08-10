import torch

from asr.modules.conformer import FastConformerEncoder


def test_fast_conformer_encoder_preserves_chunkwise_streaming_context() -> None:
    torch.manual_seed(0)
    encoder = FastConformerEncoder(
        input_size=16,
        hidden_size=8,
        num_heads=2,
        kernel_size=5,
        num_blocks=2,
        dropout_rate=0.0,
        min_chunk_size=1,
        max_chunk_size=4,
        conv_channels=4,
    ).eval()
    features = torch.randn(2, 25, 16, requires_grad=True)
    feature_lengths = torch.tensor([25, 17])

    outputs, output_lengths = encoder(features, feature_lengths, chunk_size=2)

    assert outputs.shape == (2, 4, 8)
    torch.testing.assert_close(output_lengths, torch.tensor([4, 3]))
    assert torch.isfinite(outputs).all()
    torch.testing.assert_close(outputs[1, 3:], torch.zeros_like(outputs[1, 3:]))

    changed_future = features.detach().clone()
    changed_future[:, 9:] = torch.randn_like(changed_future[:, 9:])
    changed_outputs, _ = encoder(changed_future, feature_lengths, chunk_size=2)
    torch.testing.assert_close(outputs[:, :2], changed_outputs[:, :2])

    outputs.sum().backward()
    assert features.grad is not None
    assert all(parameter.grad is not None for parameter in encoder.parameters())


def test_fast_conformer_encoder_uses_full_context_according_to_probability() -> None:
    torch.manual_seed(0)
    encoder = FastConformerEncoder(
        input_size=16,
        hidden_size=8,
        num_heads=2,
        kernel_size=5,
        num_blocks=1,
        dropout_rate=0.0,
        min_chunk_size=1,
        max_chunk_size=2,
        streaming_mask_probability=0.0,
        conv_channels=4,
    ).train()
    features = torch.randn(2, 25, 16)
    feature_lengths = torch.tensor([25, 17])

    outputs, output_lengths = encoder(features, feature_lengths)
    expected_outputs, expected_lengths = encoder(features, feature_lengths, chunk_size=4)

    torch.testing.assert_close(outputs, expected_outputs)
    torch.testing.assert_close(output_lengths, expected_lengths)

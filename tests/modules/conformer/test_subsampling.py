import torch

from asr.modules.conformer import FastConformerSubsampling


def test_fast_conformer_subsampling_returns_causal_valid_outputs() -> None:
    torch.manual_seed(0)
    features = torch.randn(2, 17, 80)
    feature_lengths = torch.tensor([17, 10])
    subsampling = FastConformerSubsampling(input_size=80, output_size=32, conv_channels=8).eval()

    outputs, output_lengths = subsampling(features, feature_lengths)

    assert outputs.shape == (2, 3, 32)
    torch.testing.assert_close(output_lengths, torch.tensor([3, 2]))
    assert torch.isfinite(outputs).all()
    torch.testing.assert_close(outputs[1, 2:], torch.zeros_like(outputs[1, 2:]))

    changed_future = features.clone()
    changed_future[:, 9:] = torch.randn_like(changed_future[:, 9:])
    changed_outputs, _ = subsampling(changed_future, feature_lengths)
    torch.testing.assert_close(outputs[:, :2], changed_outputs[:, :2])

    cache = None
    chunk_outputs = []
    for chunk in features[:1].split((3, 4, 10), dim=1):
        chunk_output, cache = subsampling.forward_chunk(chunk, cache)
        chunk_outputs.append(chunk_output)

    torch.testing.assert_close(torch.cat(chunk_outputs, dim=1), outputs[:1])
    assert cache is not None
    assert tuple(stage.num_frames for stage in cache.stages) == (17, 9, 5)

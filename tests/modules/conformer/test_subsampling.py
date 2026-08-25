import torch

from asr.modules.conformer import CausalFastConformerSubsampling, FastConformerSubsampling


def test_fast_conformer_subsampling_returns_non_causal_valid_outputs() -> None:
    torch.manual_seed(0)
    features = torch.randn(2, 17, 80)
    feature_lengths = torch.tensor([17, 10])
    subsampling = FastConformerSubsampling(input_size=80, output_size=32, conv_channels=8).eval()

    outputs, output_lengths = subsampling(features, feature_lengths)

    assert outputs.shape == (2, 3, 32)
    torch.testing.assert_close(output_lengths, torch.tensor([3, 2]))
    assert torch.isfinite(outputs).all()
    torch.testing.assert_close(outputs[1, 2:], torch.zeros_like(outputs[1, 2:]))

    assert subsampling.input_conv.padding == (1, 1)
    assert all(convolution.padding == (1, 1) for convolution in subsampling.depthwise_convs)

    short_output, _ = subsampling(features[1:2, :10], torch.tensor([10]))
    torch.testing.assert_close(short_output, outputs[1:2, :2])


def test_causal_fast_conformer_subsampling_matches_chunked_output() -> None:
    torch.manual_seed(0)
    features = torch.randn(2, 17, 80)
    feature_lengths = torch.tensor([17, 10])
    subsampling = CausalFastConformerSubsampling(input_size=80, output_size=32, conv_channels=8).eval()

    assert isinstance(subsampling, FastConformerSubsampling)
    assert subsampling.input_conv.padding == (0, 1)
    assert all(convolution.padding == (0, 1) for convolution in subsampling.depthwise_convs)

    outputs, output_lengths = subsampling(features, feature_lengths)

    assert outputs.shape == (2, 3, 32)
    torch.testing.assert_close(output_lengths, torch.tensor([3, 2]))
    torch.testing.assert_close(outputs[1, 2:], torch.zeros_like(outputs[1, 2:]))

    changed_future = features.clone()
    changed_future[:, 9:] = torch.randn_like(changed_future[:, 9:])
    changed_outputs, _ = subsampling(changed_future, feature_lengths)
    torch.testing.assert_close(outputs[:, :2], changed_outputs[:, :2])

    for chunk_sizes in ((17,), (3, 4, 10), (1,) * 17):
        cache = None
        chunk_outputs = []
        for chunk in features[:1].split(chunk_sizes, dim=1):
            chunk_output, cache = subsampling.forward_chunk(chunk, cache)
            chunk_outputs.append(chunk_output)

        torch.testing.assert_close(torch.cat(chunk_outputs, dim=1), outputs[:1])
        assert cache is not None
        assert tuple(buffer.shape[2] for buffer in cache.buffers) == (1, 1, 1)


def test_causal_fast_conformer_subsampling_preserves_cached_dtype_after_empty_chunk() -> None:
    subsampling = CausalFastConformerSubsampling(input_size=80, output_size=32, conv_channels=8).eval()

    with torch.autocast("cpu", dtype=torch.bfloat16):
        _, cache = subsampling.forward_chunk(torch.randn(1, 64, 80))
        output, next_cache = subsampling.forward_chunk(torch.empty(1, 0, 80), cache)

    assert output.shape == (1, 0, 32)
    assert output.dtype == torch.bfloat16
    assert tuple(buffer.dtype for buffer in next_cache.buffers) == (
        torch.float32,
        torch.bfloat16,
        torch.bfloat16,
    )

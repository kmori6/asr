import torch

from asr.modules.transformer.positional_encoding import PositionalEncoding


def test_positional_encoding_offset_matches_full_sequence_and_extends() -> None:
    module = PositionalEncoding(hidden_size=5, max_length=2)
    inputs = torch.zeros(2, 5, 5, dtype=torch.float64)

    full_encoding = module(inputs)
    incremental_encoding = torch.cat(
        [module(inputs[:, position : position + 1], offset=position) for position in range(inputs.shape[1])],
        dim=1,
    )

    torch.testing.assert_close(incremental_encoding, full_encoding)
    assert full_encoding.shape == (1, 5, 5)
    assert full_encoding.dtype == inputs.dtype
    assert module.max_length >= inputs.shape[1]

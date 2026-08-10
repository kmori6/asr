import torch

from asr.modules.conformer import chunk_attention_mask


def test_chunk_attention_mask_allows_current_and_preceding_chunks() -> None:
    lengths = torch.tensor([6, 4])

    mask = chunk_attention_mask(lengths, chunk_size=2, max_length=6)

    expected = torch.tensor(
        [
            [
                [1, 1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
            ],
            [
                [1, 1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ],
        ],
        dtype=torch.bool,
    )

    assert mask.dtype == torch.bool
    torch.testing.assert_close(mask, expected)

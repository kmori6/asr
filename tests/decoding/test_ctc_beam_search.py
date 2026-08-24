import itertools
import math
from collections import defaultdict

import torch
from pytest import approx, raises

from asr.decoding import CTCBeamSearch, CTCBeamSearchResult


def _collapse_alignment(alignment: tuple[int, ...], blank_token_id: int) -> tuple[int, ...]:
    token_ids: list[int] = []
    previous_token_id: int | None = None
    for token_id in alignment:
        if token_id != blank_token_id and token_id != previous_token_id:
            token_ids.append(token_id)
        previous_token_id = token_id
    return tuple(token_ids)


def _exhaustive_best(logits: torch.Tensor, blank_token_id: int) -> tuple[list[int], float]:
    probabilities = torch.softmax(logits, dim=-1)
    sequence_probabilities: defaultdict[tuple[int, ...], float] = defaultdict(float)
    for alignment in itertools.product(range(logits.shape[1]), repeat=logits.shape[0]):
        alignment_probability = math.prod(
            float(probabilities[frame_index, token_id]) for frame_index, token_id in enumerate(alignment)
        )
        sequence_probabilities[_collapse_alignment(alignment, blank_token_id)] += alignment_probability

    best_sequence, best_probability = min(
        sequence_probabilities.items(),
        key=lambda item: (-math.log(item[1]) / max(1, len(item[0])), item[0]),
    )
    return list(best_sequence), math.log(best_probability) / max(1, len(best_sequence))


def test_ctc_beam_search_matches_exhaustive_alignment_sum() -> None:
    logits = torch.log(
        torch.tensor(
            [
                [0.10, 0.65, 0.25],
                [0.50, 0.35, 0.15],
                [0.10, 0.55, 0.35],
            ]
        )
    )
    expected_token_ids, expected_score = _exhaustive_best(logits, blank_token_id=0)

    result = CTCBeamSearch(beam_width=32, blank_token_id=0).search(logits)

    assert isinstance(result, CTCBeamSearchResult)
    assert result.token_ids == expected_token_ids
    assert result.score == approx(expected_score)


def test_ctc_beam_search_applies_transition_probabilities_once_per_output_token() -> None:
    logits = torch.log(torch.tensor([[0.10, 0.70, 0.20]]))

    result = CTCBeamSearch(
        beam_width=3,
        blank_token_id=0,
        transition_scorer=lambda _prefix, token_id: -math.inf if token_id == 1 else 0.0,
    ).search(logits)

    assert result.token_ids == [2]
    assert result.score == approx(math.log(0.20))


def test_ctc_beam_search_emits_repeated_label_only_across_blank() -> None:
    logits = torch.log(
        torch.tensor(
            [
                [0.05, 0.90, 0.05],
                [0.90, 0.05, 0.05],
                [0.05, 0.90, 0.05],
            ]
        )
    )

    result = CTCBeamSearch(beam_width=32, blank_token_id=0).search(logits)

    assert result.token_ids == [1, 1]


def test_ctc_beam_search_rejects_invalid_logits_shape() -> None:
    with raises(ValueError, match="num_frames, vocab_size"):
        CTCBeamSearch(beam_width=1, blank_token_id=0).search(torch.zeros(1, 2, 3))

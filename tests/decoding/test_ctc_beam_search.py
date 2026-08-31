import itertools
import math
from collections import defaultdict
from collections.abc import Callable

import torch
from pytest import approx, raises

from asr.decoding import CTCBeamSearch, CTCBeamSearchResult
from asr.models import TransformerLMCache
from asr.modules.transformer.cache import KVCache


class _FakeLanguageModel:
    def __init__(self, vocab_size: int = 5) -> None:
        self.vocab_size = vocab_size
        self.input_token_ids: list[int] = []

    def predict(
        self,
        input_ids: torch.Tensor,
        cache: TransformerLMCache | None = None,
    ) -> tuple[torch.Tensor, TransformerLMCache]:
        self.input_token_ids.extend(input_ids[:, 0].tolist())
        value = input_ids.float()[:, None, :, None]
        if cache is None:
            key = value
        else:
            key = torch.cat((cache.layers[0].key, value), dim=2)

        logits = torch.zeros(input_ids.shape[0], 1, self.vocab_size, device=input_ids.device)
        logits[:, :, 2] = 3.0
        return logits, TransformerLMCache(layers=(KVCache(key=key, value=key),))


def _exhaustive_best(
    logits: torch.Tensor,
    blank_token_id: int,
    transition_scorer: Callable[[tuple[int, ...], int], float] | None = None,
) -> tuple[list[int], float]:
    probabilities = torch.softmax(logits, dim=-1)
    sequence_probabilities: defaultdict[tuple[int, ...], float] = defaultdict(float)
    for alignment in itertools.product(range(logits.shape[1]), repeat=logits.shape[0]):
        alignment_probability = math.prod(
            float(probabilities[frame_index, token_id]) for frame_index, token_id in enumerate(alignment)
        )
        prefix: tuple[int, ...] = ()
        previous_token_id: int | None = None
        for token_id in alignment:
            if token_id != blank_token_id and token_id != previous_token_id:
                if transition_scorer is not None:
                    alignment_probability *= math.exp(transition_scorer(prefix, token_id))
                prefix = (*prefix, token_id)
            previous_token_id = token_id
        sequence_probabilities[prefix] += alignment_probability

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
    logits = torch.log(
        torch.tensor(
            [
                [0.10, 0.70, 0.20],
                [0.60, 0.30, 0.10],
                [0.10, 0.60, 0.30],
            ]
        )
    )

    def transition_scorer(prefix: tuple[int, ...], token_id: int) -> float:
        probability = {
            (): {1: 0.25, 2: 0.75},
            (1,): {1: 0.80, 2: 0.20},
            (2,): {1: 0.60, 2: 0.40},
        }.get(prefix, {1: 0.50, 2: 0.50})[token_id]
        return math.log(probability)

    expected_token_ids, expected_score = _exhaustive_best(
        logits,
        blank_token_id=0,
        transition_scorer=transition_scorer,
    )

    result = CTCBeamSearch(
        beam_width=32,
        blank_token_id=0,
        transition_scorer=transition_scorer,
    ).search(logits)

    assert result.token_ids == expected_token_ids
    assert result.score == approx(expected_score)


def test_ctc_beam_search_applies_language_model_as_transition_probability() -> None:
    logits = torch.log(torch.tensor([[0.10, 0.70, 0.20, 1e-5, 1e-5]]))
    language_model = _FakeLanguageModel()
    searcher = CTCBeamSearch(
        beam_width=5,
        blank_token_id=0,
        language_model=language_model,
        bos_token_id=3,
        eos_token_id=4,
    )

    without_language_model = searcher.search(logits, language_model_weight=0.0)
    with_language_model = searcher.search(logits, language_model_weight=1.0)

    assert without_language_model.token_ids == [1]
    assert with_language_model.token_ids == [2]
    assert language_model.input_token_ids == [3]


def test_ctc_beam_search_advances_language_model_only_for_retained_prefixes() -> None:
    logits = torch.log(
        torch.tensor(
            [
                [0.10, 0.60, 0.30, 1e-5, 1e-5],
                [0.10, 0.60, 0.30, 1e-5, 1e-5],
            ]
        )
    )
    language_model = _FakeLanguageModel()
    searcher = CTCBeamSearch(
        beam_width=2,
        blank_token_id=0,
        language_model=language_model,
        bos_token_id=3,
        eos_token_id=4,
    )

    searcher.search(logits, language_model_weight=1.0)

    assert language_model.input_token_ids[0] == 3
    assert set(language_model.input_token_ids[1:]).issubset({1, 2})
    assert len(language_model.input_token_ids) <= 3


def test_ctc_beam_search_requires_language_model_for_positive_weight() -> None:
    logits = torch.zeros(1, 3)

    with raises(ValueError, match="language_model must be provided"):
        CTCBeamSearch(beam_width=2, blank_token_id=0).search(logits, language_model_weight=0.2)


def test_ctc_beam_search_excludes_bos_and_eos_from_ctc_labels() -> None:
    logits = torch.tensor([[0.0, 1.0, 0.5, 20.0, 20.0]])

    result = CTCBeamSearch(
        beam_width=5,
        blank_token_id=0,
        bos_token_id=3,
        eos_token_id=4,
    ).search(logits)

    assert result.token_ids == [1]


def test_ctc_beam_search_chunked_decoding_matches_complete_logits() -> None:
    logits = torch.log(
        torch.tensor(
            [
                [0.10, 0.65, 0.25],
                [0.50, 0.35, 0.15],
                [0.10, 0.55, 0.35],
            ]
        )
    )
    searcher = CTCBeamSearch(beam_width=32, blank_token_id=0)
    expected = searcher.search(logits)

    searcher.reset()
    searcher.search_chunk(logits[:1])
    result = searcher.search_chunk(logits[1:])

    assert result.token_ids == expected.token_ids
    assert result.score == approx(expected.score)


def test_ctc_beam_search_chunked_lm_fusion_matches_complete_logits() -> None:
    logits = torch.log(
        torch.tensor(
            [
                [0.10, 0.60, 0.30, 1e-5, 1e-5],
                [0.50, 0.20, 0.30, 1e-5, 1e-5],
                [0.10, 0.60, 0.30, 1e-5, 1e-5],
            ]
        )
    )
    expected = CTCBeamSearch(
        beam_width=32,
        blank_token_id=0,
        language_model=_FakeLanguageModel(),
        bos_token_id=3,
        eos_token_id=4,
    ).search(logits, language_model_weight=0.5)
    searcher = CTCBeamSearch(
        beam_width=32,
        blank_token_id=0,
        language_model=_FakeLanguageModel(),
        bos_token_id=3,
        eos_token_id=4,
    )

    searcher.search_chunk(logits[:2], language_model_weight=0.5)
    result = searcher.search_chunk(logits[2:], language_model_weight=0.5)

    assert result.token_ids == expected.token_ids
    assert result.score == approx(expected.score)


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

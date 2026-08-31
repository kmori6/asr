import itertools

import pytest
import torch
from pytest import approx

from asr.decoding import EncoderDecoderBeamSearch, EncoderDecoderBeamSearchResult
from asr.decoding.encoder_decoder_beam_search import _CTCPrefixScorer
from asr.models import TransformerLMCache
from asr.modules.transformer.cache import DecoderLayerCache, KVCache


class _FakeModel:
    def __init__(self) -> None:
        self.blank_token_id = 2
        self.position = 0
        self.batch_sizes: list[int] = []
        self.cache_batch_sizes: list[int] = []

    def ctc_log_probs(self, encoder_outputs: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(*encoder_outputs.shape[:2], 4, device=encoder_outputs.device)
        return logits.log_softmax(dim=-1)

    def embed(self, token_ids: torch.Tensor, offset: int = 0) -> torch.Tensor:
        self.position = offset
        return token_ids.float().unsqueeze(-1)

    def predict(
        self,
        encoder_outputs: torch.Tensor,
        decoder_inputs: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        caches: list[DecoderLayerCache],
    ) -> tuple[torch.Tensor, list[DecoderLayerCache]]:
        batch_size = decoder_inputs.shape[0]
        self.batch_sizes.append(batch_size)
        assert encoder_outputs.shape[0] == batch_size
        assert encoder_attention_mask.shape[0] == batch_size

        new_value = decoder_inputs[:, None, :, :]
        if caches:
            self.cache_batch_sizes.append(caches[0].self_attention.key.shape[0])
            self_key = torch.cat((caches[0].self_attention.key, new_value), dim=2)
            cross_attention = caches[0].cross_attention
        else:
            self_key = new_value
            cross_attention = KVCache(key=new_value, value=new_value)
        cache = DecoderLayerCache(
            self_attention=KVCache(key=self_key, value=self_key),
            cross_attention=cross_attention,
        )
        return decoder_inputs, [cache]

    def logits(self, decoder_outputs: torch.Tensor) -> torch.Tensor:
        batch_size = decoder_outputs.shape[0]
        logits = torch.full((batch_size, 1, 4), -20.0, device=decoder_outputs.device)
        logits[:, :, 0] = 20.0  # BOS must not be generated after the initial position.
        if self.position == 0:
            logits[:, :, 1] = 0.0
            logits[:, :, 2] = -0.1
        elif self.position == 1:
            previous_token_ids = decoder_outputs[:, 0, 0].long()
            logits[torch.arange(batch_size), 0, previous_token_ids] = 0.0
        else:
            logits[:, :, 3] = 0.0
        return logits


class _FakeLanguageModel:
    def __init__(self, vocab_size: int = 4) -> None:
        self.vocab_size = vocab_size
        self.batch_sizes: list[int] = []
        self.cache_batch_sizes: list[int] = []

    def predict(
        self,
        input_ids: torch.Tensor,
        cache: TransformerLMCache | None = None,
    ) -> tuple[torch.Tensor, TransformerLMCache]:
        batch_size = input_ids.shape[0]
        self.batch_sizes.append(batch_size)
        new_value = input_ids.float()[:, None, :, None]
        if cache is None:
            key = new_value
            position = 0
        else:
            self.cache_batch_sizes.append(cache.layers[0].key.shape[0])
            key = torch.cat((cache.layers[0].key, new_value), dim=2)
            position = cache.layers[0].key.shape[2]

        logits = torch.zeros(batch_size, 1, self.vocab_size, device=input_ids.device)
        if position == 0:
            logits[:, :, 2] = 2.0
        return logits, TransformerLMCache(layers=(KVCache(key=key, value=key),))


class _FakeCTCModel(_FakeModel):
    def __init__(self) -> None:
        super().__init__()
        self.blank_token_id = 4

    def logits(self, decoder_outputs: torch.Tensor) -> torch.Tensor:
        batch_size = decoder_outputs.shape[0]
        logits = torch.full((batch_size, 1, 5), -20.0, device=decoder_outputs.device)
        logits[:, :, 0] = 20.0  # BOS is masked by the searcher.
        if self.position == 0:
            logits[:, :, 1] = 2.0
            logits[:, :, 2] = 1.5
        else:
            logits[:, :, 3] = 5.0
        return logits

    def ctc_log_probs(self, encoder_outputs: torch.Tensor) -> torch.Tensor:
        probabilities = torch.tensor(
            [
                [0.001, 0.050, 0.890, 0.001, 0.058],
                [0.001, 0.020, 0.200, 0.001, 0.778],
                [0.001, 0.020, 0.200, 0.001, 0.778],
            ],
            device=encoder_outputs.device,
        )
        return probabilities.log().unsqueeze(0)


def _collapse_ctc_path(path: tuple[int, ...], blank_token_id: int) -> tuple[int, ...]:
    collapsed = []
    previous_token_id = blank_token_id
    for token_id in path:
        if token_id != blank_token_id and token_id != previous_token_id:
            collapsed.append(token_id)
        previous_token_id = token_id
    return tuple(collapsed)


def _ctc_sequence_probability(
    probabilities: torch.Tensor,
    prefix: tuple[int, ...],
    blank_token_id: int,
    exact: bool = False,
) -> float:
    total = 0.0
    for path in itertools.product(range(probabilities.shape[1]), repeat=probabilities.shape[0]):
        sequence = _collapse_ctc_path(path, blank_token_id)
        matches = sequence == prefix if exact else sequence[: len(prefix)] == prefix
        if matches:
            path_probability = 1.0
            for frame, token_id in enumerate(path):
                path_probability *= float(probabilities[frame, token_id])
            total += path_probability
    return total


def test_encoder_decoder_beam_search_batches_beams_and_reorders_caches() -> None:
    model = _FakeModel()
    searcher = EncoderDecoderBeamSearch(model, bos_token_id=0, eos_token_id=3)

    result = searcher.search(
        encoder_outputs=torch.zeros(1, 2, 4),
        encoder_attention_mask=torch.tensor([[True, False]]),
        beam_size=2,
        max_new_tokens=4,
        length_penalty=0.6,
    )

    assert isinstance(result, EncoderDecoderBeamSearchResult)
    assert result.token_ids[0] == 0
    assert result.token_ids[-1] == 3
    assert model.batch_sizes == [1, 2, 2]
    assert model.cache_batch_sizes == [2, 2]


def test_encoder_decoder_beam_search_uses_gnmt_length_penalty() -> None:
    assert EncoderDecoderBeamSearch._length_penalty(1, 0.6) == approx(1.0)
    assert EncoderDecoderBeamSearch._length_penalty(7, 0.6) == approx(2.0**0.6)


def test_encoder_decoder_beam_search_applies_lm_shallow_fusion_and_reorders_cache() -> None:
    model = _FakeModel()
    language_model = _FakeLanguageModel()
    searcher = EncoderDecoderBeamSearch(
        model,
        bos_token_id=0,
        eos_token_id=3,
        language_model=language_model,
    )

    result = searcher.search(
        encoder_outputs=torch.zeros(1, 2, 4),
        encoder_attention_mask=torch.tensor([[True, False]]),
        beam_size=2,
        max_new_tokens=4,
        length_penalty=0.6,
        language_model_weight=1.0,
    )

    assert result.token_ids == [0, 2, 2, 3]
    assert language_model.batch_sizes == [1, 2, 2]
    assert language_model.cache_batch_sizes == [2, 2]


def test_encoder_decoder_beam_search_requires_lm_for_positive_weight() -> None:
    searcher = EncoderDecoderBeamSearch(_FakeModel(), bos_token_id=0, eos_token_id=3)

    with pytest.raises(ValueError, match="language_model must be provided"):
        searcher.search(
            encoder_outputs=torch.zeros(1, 2, 4),
            encoder_attention_mask=torch.tensor([[True, False]]),
            beam_size=2,
            max_new_tokens=4,
            length_penalty=0.6,
            language_model_weight=0.2,
        )


def test_ctc_prefix_scorer_matches_exhaustive_path_probabilities() -> None:
    probabilities = torch.tensor(
        [
            [0.05, 0.30, 0.20, 0.05, 0.40],
            [0.05, 0.20, 0.30, 0.05, 0.40],
            [0.05, 0.25, 0.25, 0.05, 0.40],
        ]
    )
    scorer = _CTCPrefixScorer(
        probabilities.log(),
        blank_token_id=4,
        bos_token_id=0,
        eos_token_id=3,
    )
    initial_state = scorer.initial_state()
    first_scores, first_states = scorer.score_extensions(torch.tensor([0]), initial_state, initial=True)

    assert first_scores[0, 1].exp().item() == approx(
        _ctc_sequence_probability(probabilities, (1,), blank_token_id=4),
        rel=1e-5,
    )
    second_scores, _ = scorer.score_extensions(
        torch.tensor([1]),
        first_states[:, 1],
    )
    assert second_scores[0, 1].exp().item() == approx(
        _ctc_sequence_probability(probabilities, (1, 1), blank_token_id=4),
        rel=1e-5,
    )
    assert second_scores[0, 2].exp().item() == approx(
        _ctc_sequence_probability(probabilities, (1, 2), blank_token_id=4),
        rel=1e-5,
    )
    assert second_scores[0, 3].exp().item() == approx(
        _ctc_sequence_probability(probabilities, (1,), blank_token_id=4, exact=True),
        rel=1e-5,
    )
    assert torch.isneginf(second_scores[0, 0])
    assert torch.isneginf(second_scores[0, 4])


def test_encoder_decoder_beam_search_ctc_weight_zero_preserves_attention_only_search() -> None:
    language_model = _FakeLanguageModel(vocab_size=5)
    encoder_outputs = torch.zeros(1, 3, 4)
    encoder_attention_mask = torch.ones(1, 3, dtype=torch.bool)
    expected = EncoderDecoderBeamSearch(
        _FakeCTCModel(),
        bos_token_id=0,
        eos_token_id=3,
        language_model=language_model,
    ).search(
        encoder_outputs,
        encoder_attention_mask,
        beam_size=2,
        max_new_tokens=3,
        length_penalty=0.6,
        language_model_weight=0.2,
    )
    actual = EncoderDecoderBeamSearch(
        _FakeCTCModel(),
        bos_token_id=0,
        eos_token_id=3,
        language_model=_FakeLanguageModel(vocab_size=5),
    ).search(
        encoder_outputs,
        encoder_attention_mask,
        beam_size=2,
        max_new_tokens=3,
        length_penalty=0.6,
        language_model_weight=0.2,
        ctc_weight=0.0,
    )

    assert actual == expected


def test_encoder_decoder_beam_search_combines_ctc_attention_and_lm_scores() -> None:
    model = _FakeCTCModel()
    language_model = _FakeLanguageModel(vocab_size=5)
    searcher = EncoderDecoderBeamSearch(
        model,
        bos_token_id=0,
        eos_token_id=3,
        language_model=language_model,
    )

    result = searcher.search(
        encoder_outputs=torch.zeros(1, 3, 4),
        encoder_attention_mask=torch.ones(1, 3, dtype=torch.bool),
        beam_size=2,
        max_new_tokens=3,
        length_penalty=0.6,
        language_model_weight=0.2,
        ctc_weight=0.8,
    )

    assert result.token_ids == [0, 2, 3]
    assert model.batch_sizes == [1, 2]
    assert language_model.batch_sizes == [1, 2]
    assert language_model.cache_batch_sizes == [2]


def test_encoder_decoder_beam_search_validates_ctc_weight() -> None:
    searcher = EncoderDecoderBeamSearch(
        _FakeCTCModel(),
        bos_token_id=0,
        eos_token_id=3,
    )

    with pytest.raises(ValueError, match="ctc_weight must be in"):
        searcher.search(
            encoder_outputs=torch.zeros(1, 3, 4),
            encoder_attention_mask=torch.ones(1, 3, dtype=torch.bool),
            beam_size=2,
            max_new_tokens=3,
            length_penalty=0.6,
            ctc_weight=1.1,
        )


def test_encoder_decoder_beam_search_supports_ctc_only_score() -> None:
    searcher = EncoderDecoderBeamSearch(
        _FakeCTCModel(),
        bos_token_id=0,
        eos_token_id=3,
    )

    result = searcher.search(
        encoder_outputs=torch.zeros(1, 3, 4),
        encoder_attention_mask=torch.ones(1, 3, dtype=torch.bool),
        beam_size=2,
        max_new_tokens=3,
        length_penalty=0.6,
        ctc_weight=1.0,
    )

    assert result.token_ids == [0, 2, 3]
    assert result.score > -torch.inf

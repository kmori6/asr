import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch

from asr.models.transformer_lm import TransformerLMCache

CTCTransitionScorer = Callable[[tuple[int, ...], int], float]
_LOG_ZERO = -math.inf


@dataclass(frozen=True, slots=True)
class CTCBeamSearchResult:
    """Best CTC prefix and its length-normalized log-score."""

    token_ids: list[int]
    score: float


@dataclass(frozen=True, slots=True)
class _PrefixScores:
    blank: float
    non_blank: float

    @property
    def total(self) -> float:
        return _log_add(self.blank, self.non_blank)


class CTCLanguageModel(Protocol):
    """Causal language-model operation required by CTC beam search."""

    def predict(
        self,
        input_ids: torch.Tensor,
        cache: TransformerLMCache | None = None,
    ) -> tuple[torch.Tensor, TransformerLMCache]: ...


@dataclass(frozen=True, slots=True)
class _LanguageModelState:
    cache: TransformerLMCache
    next_token_log_probabilities: tuple[float, ...]


@dataclass(slots=True)
class _SearchState:
    beam: dict[tuple[int, ...], _PrefixScores]
    language_model_states: dict[tuple[int, ...], _LanguageModelState]
    vocab_size: int
    excluded_token_ids: set[int]
    device: torch.device
    language_model_weight: float


def _log_add(first: float, second: float) -> float:
    """Add two log-domain probabilities without leaving Python scalars."""
    if first == _LOG_ZERO:
        return second
    if second == _LOG_ZERO:
        return first
    maximum = max(first, second)
    return maximum + math.log1p(math.exp(-abs(first - second)))


class CTCBeamSearch:
    """Proposed in A. Graves and N. Jaitly, "Towards End-to-End Speech Recognition
    with Recurrent Neural Networks," ICML 2014.

    """

    def __init__(
        self,
        beam_width: int,
        blank_token_id: int,
        transition_scorer: CTCTransitionScorer | None = None,
        language_model: CTCLanguageModel | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
    ) -> None:
        if beam_width < 1:
            raise ValueError(f"beam_width must be >= 1, but got {beam_width}")
        if blank_token_id < 0:
            raise ValueError("blank_token_id must be non-negative")
        if (bos_token_id is None) != (eos_token_id is None):
            raise ValueError("bos_token_id and eos_token_id must either both be set or both be omitted")
        if bos_token_id is not None:
            assert eos_token_id is not None
            if bos_token_id < 0 or eos_token_id < 0:
                raise ValueError("bos_token_id and eos_token_id must be non-negative")
            if len({blank_token_id, bos_token_id, eos_token_id}) != 3:
                raise ValueError("blank_token_id, bos_token_id, and eos_token_id must be different")
        if language_model is not None and bos_token_id is None:
            raise ValueError("bos_token_id and eos_token_id must be set when language_model is provided")

        self.beam_width = beam_width
        self.blank_token_id = blank_token_id
        self.transition_scorer = transition_scorer
        self.language_model = language_model
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self._streaming_state: _SearchState | None = None

    @torch.inference_mode()
    def search(self, logits: torch.Tensor, language_model_weight: float = 0.0) -> CTCBeamSearchResult:
        """Decode frame logits for one utterance.

        Args:
            logits: Finite floating-point tensor with shape
                ``(num_frames, vocab_size)``. The vocabulary must contain the
                configured blank and at least one non-blank token.
            language_model_weight: Weight applied to ``log Pr(k | y)`` from
                the language model. Zero disables language-model scoring.

        Returns:
            The prefix maximizing the paper's final length-normalized score.
            ``score`` is its equivalent normalized log-score.
        """
        state = self._initial_state(logits, language_model_weight)
        frame_log_probabilities = torch.log_softmax(logits.float(), dim=-1).tolist()
        self._advance(state, frame_log_probabilities)
        return self._best_result(state.beam)

    def reset(self) -> None:
        """Clear prefix and language-model states from streaming decoding."""
        self._streaming_state = None

    @torch.inference_mode()
    def search_chunk(self, logits: torch.Tensor, language_model_weight: float = 0.0) -> CTCBeamSearchResult:
        """Decode the next frame-logit chunk while retaining prefix and LM states."""
        if self._streaming_state is None:
            self._streaming_state = self._initial_state(logits, language_model_weight)
        else:
            self._validate_logits(logits)
            state = self._streaming_state
            if logits.shape[1] != state.vocab_size:
                raise ValueError("vocab_size must remain fixed while decoding one utterance")
            if logits.device != state.device:
                raise ValueError("logit chunks must remain on the same device while decoding one utterance")
            if language_model_weight != state.language_model_weight:
                raise ValueError("language_model_weight must remain fixed while decoding one utterance")

        frame_log_probabilities = torch.log_softmax(logits.float(), dim=-1).tolist()
        self._advance(self._streaming_state, frame_log_probabilities)
        return self._best_result(self._streaming_state.beam)

    def _initial_state(self, logits: torch.Tensor, language_model_weight: float) -> _SearchState:
        self._validate_logits(logits)
        if language_model_weight < 0.0:
            raise ValueError("language_model_weight must be non-negative")
        if language_model_weight > 0.0 and self.language_model is None:
            raise ValueError("language_model must be provided when language_model_weight is positive")

        vocab_size = logits.shape[1]
        excluded_token_ids = {self.blank_token_id}
        if self.bos_token_id is not None:
            assert self.eos_token_id is not None
            excluded_token_ids.update((self.bos_token_id, self.eos_token_id))
        label_token_ids = [token_id for token_id in range(vocab_size) if token_id not in excluded_token_ids]
        if not label_token_ids:
            raise ValueError("the vocabulary must contain at least one non-special CTC label")

        language_model_states: dict[tuple[int, ...], _LanguageModelState] = {}
        if language_model_weight > 0.0:
            assert self.bos_token_id is not None
            language_model_states[()] = self._predict_language_model(
                self.bos_token_id,
                cache=None,
                vocab_size=vocab_size,
                excluded_token_ids=excluded_token_ids,
                device=logits.device,
            )

        return _SearchState(
            beam={(): _PrefixScores(blank=0.0, non_blank=_LOG_ZERO)},
            language_model_states=language_model_states,
            vocab_size=vocab_size,
            excluded_token_ids=excluded_token_ids,
            device=logits.device,
            language_model_weight=language_model_weight,
        )

    def _advance(self, state: _SearchState, frame_log_probabilities: list[list[float]]) -> None:
        label_token_ids = [token_id for token_id in range(state.vocab_size) if token_id not in state.excluded_token_ids]
        for frame_log_probs in frame_log_probabilities:
            retained = dict(
                sorted(
                    state.beam.items(),
                    key=lambda item: (-item[1].total, item[0]),
                )[: self.beam_width]
            )
            if state.language_model_states:
                previous_language_model_states = state.language_model_states
                state.language_model_states = {}
                for prefix in retained:
                    language_model_state = previous_language_model_states.get(prefix)
                    if language_model_state is None:
                        parent_state = previous_language_model_states.get(prefix[:-1])
                        if parent_state is None:
                            raise RuntimeError("retained CTC prefix has no language-model parent state")
                        language_model_state = self._predict_language_model(
                            prefix[-1],
                            cache=parent_state.cache,
                            vocab_size=state.vocab_size,
                            excluded_token_ids=state.excluded_token_ids,
                            device=state.device,
                        )
                    state.language_model_states[prefix] = language_model_state

            next_blank: dict[tuple[int, ...], float] = {}
            next_non_blank: dict[tuple[int, ...], float] = {}
            for prefix, prefix_scores in retained.items():
                next_blank[prefix] = prefix_scores.total + frame_log_probs[self.blank_token_id]

                if prefix:
                    final_token_id = prefix[-1]
                    next_non_blank[prefix] = _log_add(
                        next_non_blank.get(prefix, _LOG_ZERO),
                        prefix_scores.non_blank + frame_log_probs[final_token_id],
                    )

                for token_id in label_token_ids:
                    extended_prefix = (*prefix, token_id)
                    extension_log_probability = self._extension_log_probability(
                        prefix,
                        token_id,
                        prefix_scores,
                        frame_log_probs[token_id],
                        state.language_model_states.get(prefix),
                        state.language_model_weight,
                    )
                    next_non_blank[extended_prefix] = _log_add(
                        next_non_blank.get(extended_prefix, _LOG_ZERO),
                        extension_log_probability,
                    )

            state.beam = {
                prefix: _PrefixScores(
                    blank=next_blank.get(prefix, _LOG_ZERO),
                    non_blank=next_non_blank.get(prefix, _LOG_ZERO),
                )
                for prefix in next_blank.keys() | next_non_blank.keys()
            }

    def _best_result(self, beam: dict[tuple[int, ...], _PrefixScores]) -> CTCBeamSearchResult:
        best_prefix, best_scores = min(
            beam.items(),
            key=lambda item: (-self._normalized_score(item[0], item[1]), item[0]),
        )
        return CTCBeamSearchResult(
            token_ids=list(best_prefix),
            score=self._normalized_score(best_prefix, best_scores),
        )

    def _predict_language_model(
        self,
        token_id: int,
        cache: TransformerLMCache | None,
        vocab_size: int,
        excluded_token_ids: set[int],
        device: torch.device,
    ) -> _LanguageModelState:
        assert self.language_model is not None
        input_ids = torch.tensor([[token_id]], dtype=torch.long, device=device)
        logits, next_cache = self.language_model.predict(input_ids, cache)
        if logits.shape != (1, 1, vocab_size):
            raise ValueError(f"language-model logits must have shape (1, 1, {vocab_size})")
        if logits.device != device:
            raise ValueError("language-model and CTC logits must be on the same device")

        next_token_logits = logits[0, 0].float().clone()
        next_token_logits[list(excluded_token_ids)] = -torch.inf
        next_token_log_probabilities = torch.log_softmax(next_token_logits, dim=-1)
        return _LanguageModelState(
            cache=next_cache,
            next_token_log_probabilities=tuple(next_token_log_probabilities.tolist()),
        )

    def _extension_log_probability(
        self,
        prefix: tuple[int, ...],
        token_id: int,
        prefix_scores: _PrefixScores,
        emission_log_probability: float,
        language_model_state: _LanguageModelState | None,
        language_model_weight: float,
    ) -> float:
        if prefix and prefix[-1] == token_id:
            source_log_probability = prefix_scores.blank
        else:
            source_log_probability = prefix_scores.total

        transition_log_probability = 0.0
        if self.transition_scorer is not None:
            transition_log_probability = float(self.transition_scorer(prefix, token_id))
            if math.isnan(transition_log_probability) or transition_log_probability > 0.0:
                raise ValueError("transition_scorer must return a log probability in [-inf, 0]")
        if language_model_state is not None:
            transition_log_probability += (
                language_model_weight * language_model_state.next_token_log_probabilities[token_id]
            )
        return source_log_probability + emission_log_probability + transition_log_probability

    @staticmethod
    def _normalized_score(prefix: tuple[int, ...], scores: _PrefixScores) -> float:
        return scores.total / max(1, len(prefix))

    def _validate_logits(self, logits: torch.Tensor) -> None:
        if logits.ndim != 2:
            raise ValueError(f"logits must have shape (num_frames, vocab_size), but got {tuple(logits.shape)}")
        if logits.shape[1] < 2:
            raise ValueError("vocab_size must be at least two (blank and one label)")
        if self.blank_token_id >= logits.shape[1]:
            raise ValueError(f"blank_token_id must be in [0, {logits.shape[1]}), but got {self.blank_token_id}")
        for name, token_id in (("bos_token_id", self.bos_token_id), ("eos_token_id", self.eos_token_id)):
            if token_id is not None and token_id >= logits.shape[1]:
                raise ValueError(f"{name} must be in [0, {logits.shape[1]}), but got {token_id}")
        if not logits.is_floating_point():
            raise TypeError("logits must be a floating-point tensor")
        if not torch.isfinite(logits).all():
            raise ValueError("logits must contain only finite values")

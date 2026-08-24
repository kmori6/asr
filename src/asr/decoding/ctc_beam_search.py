import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

CTCTransitionScorer = Callable[[tuple[int, ...], int], float]
_LOG_ZERO = -math.inf


@dataclass(frozen=True, slots=True)
class CTCBeamSearchResult:
    """Best CTC prefix and its length-normalized log-probability."""

    token_ids: list[int]
    score: float


@dataclass(frozen=True, slots=True)
class _PrefixScores:
    blank: float
    non_blank: float

    @property
    def total(self) -> float:
        return _log_add(self.blank, self.non_blank)


def _log_add(first: float, second: float) -> float:
    """Add two log-domain probabilities without leaving Python scalars."""
    if first == _LOG_ZERO:
        return second
    if second == _LOG_ZERO:
        return first
    maximum = max(first, second)
    return maximum + math.log1p(math.exp(-abs(first - second)))


class CTCBeamSearch:
    """Prefix beam search following Graves and Jaitly's Algorithm 1.

    Blank-ending and non-blank-ending path probabilities are accumulated
    separately so alignments that collapse to the same label prefix are merged.
    Computation uses log probabilities for numerical stability while preserving
    the probability-domain recurrences in the paper.

    Paper: https://arxiv.org/pdf/1408.2873

    Args:
        beam_width: Number of prefixes retained before each input frame.
        blank_token_id: Vocabulary index of the CTC blank symbol.
        transition_scorer: Optional function returning ``log Pr(k | y)`` for
            extending prefix ``y`` by token ``k``. When omitted, every
            transition has probability one, as in CTC decoding without a
            dictionary or language model.
    """

    def __init__(
        self,
        beam_width: int,
        blank_token_id: int,
        transition_scorer: CTCTransitionScorer | None = None,
    ) -> None:
        if beam_width < 1:
            raise ValueError(f"beam_width must be >= 1, but got {beam_width}")
        if blank_token_id < 0:
            raise ValueError("blank_token_id must be non-negative")

        self.beam_width = beam_width
        self.blank_token_id = blank_token_id
        self.transition_scorer = transition_scorer

    @torch.inference_mode()
    def search(self, logits: torch.Tensor) -> CTCBeamSearchResult:
        """Decode frame logits for one utterance.

        Args:
            logits: Finite floating-point tensor with shape
                ``(num_frames, vocab_size)``. The vocabulary must contain the
                configured blank and at least one non-blank token.

        Returns:
            The prefix maximizing the paper's final length-normalized
            probability. ``score`` is its equivalent normalized log-probability.
        """
        self._validate_logits(logits)
        frame_log_probabilities = torch.log_softmax(logits.float(), dim=-1).tolist()
        vocab_size = logits.shape[1]
        label_token_ids = [token_id for token_id in range(vocab_size) if token_id != self.blank_token_id]

        beam: dict[tuple[int, ...], _PrefixScores] = {(): _PrefixScores(blank=0.0, non_blank=_LOG_ZERO)}
        for frame_log_probs in frame_log_probabilities:
            retained = dict(
                sorted(
                    beam.items(),
                    key=lambda item: (-item[1].total, item[0]),
                )[: self.beam_width]
            )
            candidates = set(retained)
            candidates.update((*prefix, token_id) for prefix in retained for token_id in label_token_ids)

            next_beam: dict[tuple[int, ...], _PrefixScores] = {}
            for prefix in sorted(candidates):
                previous_scores = retained.get(prefix)
                blank_log_probability = (
                    previous_scores.total + frame_log_probs[self.blank_token_id]
                    if previous_scores is not None
                    else _LOG_ZERO
                )

                non_blank_log_probability = _LOG_ZERO
                if prefix:
                    final_token_id = prefix[-1]
                    if previous_scores is not None:
                        repeated_log_probability = previous_scores.non_blank + frame_log_probs[final_token_id]
                        non_blank_log_probability = _log_add(
                            non_blank_log_probability,
                            repeated_log_probability,
                        )

                    parent = prefix[:-1]
                    parent_scores = retained.get(parent)
                    if parent_scores is not None:
                        extension_log_probability = self._extension_log_probability(
                            parent,
                            final_token_id,
                            parent_scores,
                            frame_log_probs[final_token_id],
                        )
                        non_blank_log_probability = _log_add(
                            non_blank_log_probability,
                            extension_log_probability,
                        )

                next_beam[prefix] = _PrefixScores(
                    blank=blank_log_probability,
                    non_blank=non_blank_log_probability,
                )
            beam = next_beam

        best_prefix, best_scores = min(
            beam.items(),
            key=lambda item: (-self._normalized_score(item[0], item[1]), item[0]),
        )
        return CTCBeamSearchResult(
            token_ids=list(best_prefix),
            score=self._normalized_score(best_prefix, best_scores),
        )

    def _extension_log_probability(
        self,
        prefix: tuple[int, ...],
        token_id: int,
        prefix_scores: _PrefixScores,
        emission_log_probability: float,
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
        if not logits.is_floating_point():
            raise TypeError("logits must be a floating-point tensor")
        if not torch.isfinite(logits).all():
            raise ValueError("logits must contain only finite values")

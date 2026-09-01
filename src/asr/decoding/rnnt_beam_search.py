from dataclasses import dataclass
from typing import Protocol

import torch

from asr.models.transformer_lm import TransformerLMCache
from asr.modules.rnnt import JointNetwork, PredictionNetwork
from asr.modules.rnnt.prediction_network import PredictionState


@dataclass(frozen=True, slots=True)
class RNNTBeamSearchResult:
    """Best token sequence and its length-normalized decoding score."""

    token_ids: list[int]
    score: float


@dataclass(frozen=True, slots=True)
class _Hypothesis:
    token_ids: tuple[int, ...]
    score: float


class RNNTLanguageModel(Protocol):
    """Causal language-model operation required for shallow fusion."""

    def predict(
        self,
        input_ids: torch.Tensor,
        cache: TransformerLMCache | None = None,
    ) -> tuple[torch.Tensor, TransformerLMCache]: ...


@dataclass(frozen=True, slots=True)
class _LanguageModelState:
    cache: TransformerLMCache
    next_token_log_probabilities: tuple[float, ...]


class RNNTBeamSearch:
    """Beam search decoder for RNN-T models.

    Proposed in A. Graves, "Sequence transduction with recurrent neural networks,"
    arXiv preprint arXiv:1211.3711, 2012.

    """

    def __init__(
        self,
        prediction_network: PredictionNetwork,
        joint_network: JointNetwork,
        beam_width: int,
        blank_token_id: int,
        language_model: RNNTLanguageModel | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
    ) -> None:
        if beam_width < 1:
            raise ValueError(f"beam_width must be >= 1, but got {beam_width}")
        if prediction_network.vocab_size < 2:
            raise ValueError("vocab_size must be >= 2 (at least blank and one label)")
        if joint_network.vocab_size != prediction_network.vocab_size:
            raise ValueError("joint and prediction network vocabulary sizes must match")
        if joint_network.predictor_size != prediction_network.hidden_size:
            raise ValueError("joint predictor_size must match the prediction network hidden_size")
        if blank_token_id != prediction_network.blank_token_id:
            raise ValueError("blank_token_id must match the prediction network blank_token_id")
        if (bos_token_id is None) != (eos_token_id is None):
            raise ValueError("bos_token_id and eos_token_id must either both be set or both be omitted")
        if bos_token_id is not None:
            assert eos_token_id is not None
            if any(
                token_id < 0 or token_id >= prediction_network.vocab_size for token_id in (bos_token_id, eos_token_id)
            ):
                raise ValueError("bos_token_id and eos_token_id must be valid vocabulary indices")
            if len({blank_token_id, bos_token_id, eos_token_id}) != 3:
                raise ValueError("blank_token_id, bos_token_id, and eos_token_id must be different")
        if language_model is not None and bos_token_id is None:
            raise ValueError("bos_token_id and eos_token_id must be set when language_model is provided")

        self.prediction_network = prediction_network
        self.joint_network = joint_network
        self.beam_width = beam_width
        self.blank_token_id = blank_token_id
        self.language_model = language_model
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self._beam: list[_Hypothesis] = []
        self._prediction_cache: dict[tuple[int, ...], tuple[torch.Tensor, PredictionState]] = {}
        self._language_model_cache: dict[tuple[int, ...], _LanguageModelState] = {}
        self._language_model_weight: float | None = None

    def reset(self) -> None:
        """Discard all hypotheses, prediction states, and language-model states."""
        self._beam.clear()
        self._prediction_cache.clear()
        self._language_model_cache.clear()
        self._language_model_weight = None

    @torch.inference_mode()
    def search(self, encoder_outputs: torch.Tensor, language_model_weight: float = 0.0) -> RNNTBeamSearchResult:
        """
        Args:
            encoder_outputs: Encoder representations for one utterance or its
                next chunk with shape ``(1, num_frames, encoder_size)``. Calling
                this method repeatedly without :meth:`reset` continues decoding
                the same utterance.
            language_model_weight: Weight applied to the external language-model
                log-probability for non-blank extensions. Zero disables fusion.

        Returns:
            Best current hypothesis. ``token_ids`` excludes the initial blank,
            and ``score`` is the combined RNN-T and LM log-score divided by the
            number of output tokens, with an empty sequence treated as length one.
        """
        self._validate_encoder_outputs(encoder_outputs)
        if language_model_weight < 0.0:
            raise ValueError("language_model_weight must be non-negative")
        if language_model_weight > 0.0 and self.language_model is None:
            raise ValueError("language_model must be provided when language_model_weight is positive")
        if self._language_model_weight is None:
            self._language_model_weight = language_model_weight
        elif language_model_weight != self._language_model_weight:
            raise ValueError("language_model_weight must remain fixed while decoding one utterance")
        if not self._beam:
            self._beam = [_Hypothesis(token_ids=(), score=0.0)]

        excluded_token_ids = {self.blank_token_id}
        if self.bos_token_id is not None:
            assert self.eos_token_id is not None
            excluded_token_ids.update((self.bos_token_id, self.eos_token_id))
        label_token_ids = [
            token_id for token_id in range(self.prediction_network.vocab_size) if token_id not in excluded_token_ids
        ]
        if not label_token_ids:
            raise ValueError("the vocabulary must contain at least one non-special RNN-T label")

        for encoder_frame in encoder_outputs.unbind(dim=1):
            active_hypotheses = self._beam.copy()
            completed_hypotheses: list[_Hypothesis] = []

            # NOTE: The prefix-search correction from Graves' Algorithm 1 is
            # intentionally omitted following Section 5.1 of
            # https://arxiv.org/pdf/2201.05420.

            while True:
                best_index = max(range(len(active_hypotheses)), key=lambda index: active_hypotheses[index].score)
                hypothesis = active_hypotheses.pop(best_index)
                predictor_output, _ = self._prediction(hypothesis.token_ids, encoder_outputs.device)
                logits = self.joint_network(encoder_frame[:, None, :], predictor_output)
                log_probs = torch.log_softmax(logits[0, 0, 0].float(), dim=-1)
                log_prob_values = log_probs.tolist()
                language_model_state = (
                    self._language_model_prediction(hypothesis.token_ids, encoder_outputs.device)
                    if language_model_weight > 0.0
                    else None
                )

                completed_hypotheses.append(
                    _Hypothesis(
                        token_ids=hypothesis.token_ids,
                        score=hypothesis.score + log_prob_values[self.blank_token_id],
                    )
                )

                # Shallow fusion scores labels but leaves blank transitions unchanged.
                # https://arxiv.org/abs/2110.06841
                active_hypotheses.extend(
                    _Hypothesis(
                        token_ids=(*hypothesis.token_ids, token_id),
                        score=(
                            hypothesis.score
                            + log_prob_values[token_id]
                            + (
                                language_model_weight * language_model_state.next_token_log_probabilities[token_id]
                                if language_model_state is not None
                                else 0.0
                            )
                        ),
                    )
                    for token_id in label_token_ids
                )

                best_active_score = max(hypothesis.score for hypothesis in active_hypotheses)
                completed_more_probable = [
                    hypothesis for hypothesis in completed_hypotheses if hypothesis.score > best_active_score
                ]
                if len(completed_more_probable) >= self.beam_width:
                    self._beam = sorted(
                        completed_more_probable,
                        key=lambda hypothesis: hypothesis.score,
                        reverse=True,
                    )[: self.beam_width]
                    break

        best_hypothesis = max(
            self._beam,
            key=lambda hypothesis: hypothesis.score / max(1, len(hypothesis.token_ids)),
        )
        normalized_score = best_hypothesis.score / max(1, len(best_hypothesis.token_ids))
        return RNNTBeamSearchResult(token_ids=list(best_hypothesis.token_ids), score=normalized_score)

    def _prediction(
        self,
        token_ids: tuple[int, ...],
        device: torch.device,
    ) -> tuple[torch.Tensor, PredictionState]:
        cached_prediction = self._prediction_cache.get(token_ids)
        if cached_prediction is not None:
            return cached_prediction

        if token_ids:
            parent_prediction = self._prediction_cache.get(token_ids[:-1])
            if parent_prediction is None:
                raise RuntimeError("parent prediction state is missing from the cache")
            token_id = token_ids[-1]
            state = parent_prediction[1]
        else:
            token_id = self.blank_token_id
            state = None

        token = torch.tensor([[token_id]], dtype=torch.long, device=device)
        prediction = self.prediction_network(token, state)
        self._prediction_cache[token_ids] = prediction
        return prediction

    def _language_model_prediction(
        self,
        token_ids: tuple[int, ...],
        device: torch.device,
    ) -> _LanguageModelState:
        cached_prediction = self._language_model_cache.get(token_ids)
        if cached_prediction is not None:
            return cached_prediction

        assert self.language_model is not None
        assert self.bos_token_id is not None
        assert self.eos_token_id is not None
        if token_ids:
            parent_prediction = self._language_model_cache.get(token_ids[:-1])
            if parent_prediction is None:
                raise RuntimeError("parent language-model state is missing from the cache")
            token_id = token_ids[-1]
            cache = parent_prediction.cache
        else:
            token_id = self.bos_token_id
            cache = None

        input_ids = torch.tensor([[token_id]], dtype=torch.long, device=device)
        logits, next_cache = self.language_model.predict(input_ids, cache)
        vocab_size = self.prediction_network.vocab_size
        if logits.shape != (1, 1, vocab_size):
            raise ValueError(f"language-model logits must have shape (1, 1, {vocab_size})")
        if logits.device != device:
            raise ValueError("language-model and RNN-T tensors must be on the same device")

        next_token_logits = logits[0, 0].float().clone()
        next_token_logits[[self.blank_token_id, self.bos_token_id, self.eos_token_id]] = -torch.inf
        state = _LanguageModelState(
            cache=next_cache,
            next_token_log_probabilities=tuple(torch.log_softmax(next_token_logits, dim=-1).tolist()),
        )
        self._language_model_cache[token_ids] = state
        return state

    def _validate_encoder_outputs(self, encoder_outputs: torch.Tensor) -> None:
        if encoder_outputs.ndim != 3 or encoder_outputs.shape[0] != 1:
            raise ValueError(
                f"encoder_outputs must have shape (1, num_frames, encoder_size), but got {tuple(encoder_outputs.shape)}"
            )
        if encoder_outputs.shape[-1] != self.joint_network.encoder_size:
            raise ValueError(
                f"expected encoder_size {self.joint_network.encoder_size}, but got {encoder_outputs.shape[-1]}"
            )
        prediction_device = next(self.prediction_network.parameters()).device
        joint_device = next(self.joint_network.parameters()).device
        if encoder_outputs.device != prediction_device or encoder_outputs.device != joint_device:
            raise ValueError("encoder outputs, prediction network, and joint network must be on the same device")
        if self._prediction_cache:
            cached_device = next(iter(self._prediction_cache.values()))[0].device
            if encoder_outputs.device != cached_device:
                raise ValueError("encoder chunks for one utterance must remain on the same device")

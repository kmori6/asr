from dataclasses import dataclass

import torch

from asr.modules.rnnt import JointNetwork, PredictionNetwork
from asr.modules.rnnt.prediction_network import PredictionState


@dataclass(frozen=True, slots=True)
class RNNTBeamSearchResult:
    """Best token sequence and its length-normalized log-probability."""

    token_ids: list[int]
    score: float


@dataclass(frozen=True, slots=True)
class _Hypothesis:
    token_ids: tuple[int, ...]
    score: float


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

        self.prediction_network = prediction_network
        self.joint_network = joint_network
        self.beam_width = beam_width
        self.blank_token_id = blank_token_id
        self._beam: list[_Hypothesis] = []
        self._prediction_cache: dict[tuple[int, ...], tuple[torch.Tensor, PredictionState]] = {}

    def reset(self) -> None:
        """Discard all hypotheses and prediction-network states."""
        self._beam.clear()
        self._prediction_cache.clear()

    @torch.inference_mode()
    def search(self, encoder_outputs: torch.Tensor) -> RNNTBeamSearchResult:
        """
        Args:
            encoder_outputs: Encoder representations for one utterance or its
                next chunk with shape ``(1, num_frames, encoder_size)``. Calling
                this method repeatedly without :meth:`reset` continues decoding
                the same utterance.

        Returns:
            Best current hypothesis. ``token_ids`` excludes the initial blank,
            and ``score`` is the log-probability divided by the number of output
            tokens, with an empty sequence treated as length one.
        """
        self._validate_encoder_outputs(encoder_outputs)
        if not self._beam:
            self._beam = [_Hypothesis(token_ids=(), score=0.0)]

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

                completed_hypotheses.append(
                    _Hypothesis(
                        token_ids=hypothesis.token_ids,
                        score=hypothesis.score + log_prob_values[self.blank_token_id],
                    )
                )

                active_hypotheses.extend(
                    _Hypothesis(
                        token_ids=(*hypothesis.token_ids, token_id),
                        score=hypothesis.score + log_prob_values[token_id],
                    )
                    for token_id in range(self.prediction_network.vocab_size)
                    if token_id != self.blank_token_id
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

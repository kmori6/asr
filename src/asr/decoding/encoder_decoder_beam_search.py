from dataclasses import dataclass
from typing import Protocol

import torch

from asr.models.transformer_lm import TransformerLMCache
from asr.modules.transformer.cache import DecoderLayerCache, KVCache


@dataclass(frozen=True, slots=True)
class EncoderDecoderBeamSearchResult:
    """Best token sequence and its length-normalized log-probability."""

    token_ids: list[int]
    score: float


class EncoderDecoderBeamSearchModel(Protocol):
    """Model operations required for autoregressive beam search."""

    blank_token_id: int

    def ctc_log_probs(self, encoder_outputs: torch.Tensor) -> torch.Tensor: ...

    def embed(self, token_ids: torch.Tensor, offset: int = 0) -> torch.Tensor: ...

    def logits(self, decoder_outputs: torch.Tensor) -> torch.Tensor: ...

    def predict(
        self,
        encoder_outputs: torch.Tensor,
        decoder_inputs: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        caches: list[DecoderLayerCache],
    ) -> tuple[torch.Tensor, list[DecoderLayerCache]]: ...


class EncoderDecoderLanguageModel(Protocol):
    """Causal language-model operation required for shallow fusion."""

    def predict(
        self,
        input_ids: torch.Tensor,
        cache: TransformerLMCache | None = None,
    ) -> tuple[torch.Tensor, TransformerLMCache]: ...


class _CTCPrefixScorer:
    """Log-domain implementation of the paper's CTC prefix recursion."""

    _BLANK = 0
    _NON_BLANK = 1

    def __init__(
        self,
        log_probs: torch.Tensor,
        blank_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
    ) -> None:
        if log_probs.ndim != 2 or log_probs.shape[0] == 0:
            raise ValueError("CTC log probabilities must have shape (num_frames, vocab_size)")
        if not log_probs.is_floating_point():
            raise TypeError("CTC log probabilities must be floating point")
        if torch.isnan(log_probs).any() or torch.isposinf(log_probs).any():
            raise ValueError("CTC log probabilities must not contain NaN or positive infinity")
        vocab_size = log_probs.shape[1]
        if any(token_id < 0 or token_id >= vocab_size for token_id in (blank_token_id, bos_token_id, eos_token_id)):
            raise ValueError("blank, BOS, and EOS token IDs must be valid CTC vocabulary indices")
        if len({blank_token_id, bos_token_id, eos_token_id}) != 3:
            raise ValueError("blank, BOS, and EOS token IDs must be different")

        self.log_probs = log_probs.float()
        self.blank_token_id = blank_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    def initial_state(self) -> torch.Tensor:
        """Return gamma for the empty CTC prefix in Eqs. (51)-(52)."""
        num_frames = self.log_probs.shape[0]
        state = torch.full(
            (1, num_frames, 2),
            -torch.inf,
            dtype=self.log_probs.dtype,
            device=self.log_probs.device,
        )
        state[0, :, self._BLANK] = self.log_probs[:, self.blank_token_id].cumsum(dim=0)
        return state

    def score_extensions(
        self,
        last_token_ids: torch.Tensor,
        states: torch.Tensor,
        initial: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score every one-token extension using Algorithm 2.

        Args:
            last_token_ids: Last label of each active prefix, shaped ``(beam,)``.
            states: CTC blank/non-blank forward scores, shaped ``(beam, num_frames, 2)``.
            initial: Whether the active prefix is the initial BOS-only prefix.

        Returns:
            CTC prefix scores shaped ``(beam, vocab_size)`` and the corresponding
            child forward states shaped ``(beam, vocab_size, num_frames, 2)``.
        """
        num_beams = last_token_ids.shape[0]
        num_frames, vocab_size = self.log_probs.shape
        if last_token_ids.ndim != 1 or last_token_ids.dtype != torch.long:
            raise ValueError("last_token_ids must have shape (beam,) and dtype torch.long")
        if states.shape != (num_beams, num_frames, 2):
            raise ValueError(f"states must have shape ({num_beams}, {num_frames}, 2)")
        if last_token_ids.device != self.log_probs.device or states.device != self.log_probs.device:
            raise ValueError("CTC labels, states, and log probabilities must be on the same device")

        child_states = torch.full(
            (num_beams, vocab_size, num_frames, 2),
            -torch.inf,
            dtype=self.log_probs.dtype,
            device=self.log_probs.device,
        )
        child_non_blank = child_states[..., self._NON_BLANK]
        child_blank = child_states[..., self._BLANK]

        if initial:
            child_non_blank[:, :, 0] = self.log_probs[0]
        prefix_scores = child_non_blank[:, :, 0].clone()

        token_ids = torch.arange(vocab_size, device=self.log_probs.device)
        repeats_last_token = last_token_ids[:, None].eq(token_ids[None, :])
        for frame in range(1, num_frames):
            parent_blank = states[:, frame - 1, self._BLANK, None].expand(-1, vocab_size)
            parent_non_blank = states[:, frame - 1, self._NON_BLANK, None].expand(-1, vocab_size)
            parent_non_blank = parent_non_blank.masked_fill(repeats_last_token, -torch.inf)
            extension_score = torch.logaddexp(parent_blank, parent_non_blank)

            emission_scores = self.log_probs[frame][None, :]
            child_non_blank[:, :, frame] = (
                torch.logaddexp(child_non_blank[:, :, frame - 1], extension_score) + emission_scores
            )
            child_blank[:, :, frame] = (
                torch.logaddexp(child_blank[:, :, frame - 1], child_non_blank[:, :, frame - 1])
                + self.log_probs[frame, self.blank_token_id]
            )
            prefix_scores = torch.logaddexp(prefix_scores, extension_score + emission_scores)

        prefix_scores[:, self.eos_token_id] = torch.logaddexp(
            states[:, -1, self._BLANK],
            states[:, -1, self._NON_BLANK],
        )
        prefix_scores[:, self.blank_token_id] = -torch.inf
        prefix_scores[:, self.bos_token_id] = -torch.inf
        return prefix_scores, child_states


class EncoderDecoderBeamSearch:
    """Encoder-decoder beam search with optional joint CTC scoring and LM fusion.

    Proposed in S. Watanabe et al., "Hybrid CTC/Attention Architecture for End-to-End Speech Recognition,"
    in IEEE J. Sel. Topics Signal Process., 2017, pp. 1240-1253.

    """

    def __init__(
        self,
        model: EncoderDecoderBeamSearchModel,
        bos_token_id: int,
        eos_token_id: int,
        language_model: EncoderDecoderLanguageModel | None = None,
    ) -> None:
        if bos_token_id < 0 or eos_token_id < 0:
            raise ValueError("bos_token_id and eos_token_id must be non-negative")
        if bos_token_id == eos_token_id:
            raise ValueError("bos_token_id and eos_token_id must be different")
        if model.blank_token_id < 0:
            raise ValueError("blank_token_id must be non-negative")
        if model.blank_token_id in (bos_token_id, eos_token_id):
            raise ValueError("blank_token_id must be different from BOS and EOS")
        self.model = model
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.language_model = language_model
        self.blank_token_id = model.blank_token_id

    @staticmethod
    def _length_penalty(generated_length: int, alpha: float) -> float:
        # GNMT length penalty used by the Transformer paper.
        # https://arxiv.org/abs/1609.08144
        return ((5.0 + generated_length) / 6.0) ** alpha

    @staticmethod
    def _select_caches(
        caches: list[DecoderLayerCache],
        parent_indices: torch.Tensor,
    ) -> list[DecoderLayerCache]:
        """Select and reorder decoder caches by active parent beam."""
        return [
            DecoderLayerCache(
                self_attention=KVCache(
                    key=cache.self_attention.key.index_select(0, parent_indices),
                    value=cache.self_attention.value.index_select(0, parent_indices),
                ),
                cross_attention=KVCache(
                    key=cache.cross_attention.key.index_select(0, parent_indices),
                    value=cache.cross_attention.value.index_select(0, parent_indices),
                ),
            )
            for cache in caches
        ]

    @staticmethod
    def _select_language_model_cache(
        cache: TransformerLMCache,
        parent_indices: torch.Tensor,
    ) -> TransformerLMCache:
        """Select and reorder the language-model cache by active parent beam."""
        return TransformerLMCache(
            layers=tuple(
                KVCache(
                    key=layer.key.index_select(0, parent_indices),
                    value=layer.value.index_select(0, parent_indices),
                )
                for layer in cache.layers
            )
        )

    @torch.inference_mode()
    def search(
        self,
        encoder_outputs: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        beam_size: int,
        max_new_tokens: int,
        length_penalty: float,
        language_model_weight: float = 0.0,
        ctc_weight: float = 0.0,
    ) -> EncoderDecoderBeamSearchResult:
        """Decode one utterance with optional CTC and language-model scores.

        The returned sequence includes BOS and includes EOS when decoding completed. The score
        is ``(1 - ctc_weight) * attention + ctc_weight * CTC`` plus
        ``language_model_weight * LM``, followed by the length penalty. Setting either optional
        weight to zero disables that scorer.
        """
        if beam_size < 1:
            raise ValueError("beam_size must be positive")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if length_penalty < 0.0:
            raise ValueError("length_penalty must be non-negative")
        if language_model_weight < 0.0:
            raise ValueError("language_model_weight must be non-negative")
        if language_model_weight > 0.0 and self.language_model is None:
            raise ValueError("language_model must be provided when language_model_weight is positive")
        if not 0.0 <= ctc_weight <= 1.0:
            raise ValueError("ctc_weight must be in [0, 1]")
        if encoder_outputs.ndim != 3 or encoder_outputs.shape[0] != 1:
            raise ValueError("encoder_outputs must have shape (1, source_length, hidden_size)")
        expected_mask_shape = (1, encoder_outputs.shape[1])
        if encoder_attention_mask.shape != expected_mask_shape:
            raise ValueError(f"encoder_attention_mask must have shape {expected_mask_shape}")
        if encoder_attention_mask.device != encoder_outputs.device:
            raise ValueError("encoder outputs and attention mask must be on the same device")
        if not torch.any(encoder_attention_mask):
            raise ValueError("encoder_attention_mask must contain at least one valid frame")

        ctc_prefix_scorer = None
        if ctc_weight > 0.0:
            ctc_log_probs = self.model.ctc_log_probs(encoder_outputs)
            expected_ctc_shape = (1, encoder_outputs.shape[1])
            if ctc_log_probs.ndim != 3 or ctc_log_probs.shape[:2] != expected_ctc_shape:
                raise ValueError("model CTC log probabilities must have shape (1, source_length, vocab_size)")
            ctc_prefix_scorer = _CTCPrefixScorer(
                ctc_log_probs[0, encoder_attention_mask[0].bool()],
                blank_token_id=self.blank_token_id,
                bos_token_id=self.bos_token_id,
                eos_token_id=self.eos_token_id,
            )

        device = encoder_outputs.device
        sequence_capacity = max_new_tokens + 1
        alive_token_ids = torch.full(
            (1, sequence_capacity),
            self.eos_token_id,
            dtype=torch.long,
            device=device,
        )
        alive_token_ids[0, 0] = self.bos_token_id
        alive_log_probs = torch.zeros(1, dtype=torch.float32, device=device)
        alive_non_ctc_log_probs = torch.zeros(1, dtype=torch.float32, device=device)
        alive_caches: list[DecoderLayerCache] = []
        alive_language_model_cache: TransformerLMCache | None = None
        alive_ctc_states = ctc_prefix_scorer.initial_state() if ctc_prefix_scorer is not None else None

        finished_token_ids = torch.full(
            (beam_size, sequence_capacity),
            self.eos_token_id,
            dtype=torch.long,
            device=device,
        )
        finished_scores = torch.full((beam_size,), -torch.inf, dtype=torch.float32, device=device)
        finished_lengths = torch.zeros(beam_size, dtype=torch.long, device=device)

        generated_length = 0
        for position in range(max_new_tokens):
            num_alive = alive_token_ids.shape[0]
            last_token_ids = alive_token_ids[:, position : position + 1]
            decoder_inputs = self.model.embed(last_token_ids, offset=position)
            decoder_outputs, step_caches = self.model.predict(
                encoder_outputs.expand(num_alive, -1, -1),
                decoder_inputs,
                encoder_attention_mask.expand(num_alive, -1),
                alive_caches,
            )
            logits = self.model.logits(decoder_outputs)[:, -1, :].float()
            vocab_size = logits.shape[-1]
            if vocab_size <= beam_size:
                raise ValueError("vocab_size must be greater than beam_size")
            if self.bos_token_id >= vocab_size or self.eos_token_id >= vocab_size:
                raise ValueError("BOS and EOS token IDs must be valid vocabulary indices")
            logits[:, self.bos_token_id] = -torch.inf

            step_attention_log_probs = torch.log_softmax(logits, dim=-1)
            step_language_model_log_probs = None
            step_language_model_cache = None
            if language_model_weight > 0.0:
                assert self.language_model is not None
                language_model_logits, step_language_model_cache = self.language_model.predict(
                    last_token_ids,
                    alive_language_model_cache,
                )
                language_model_logits = language_model_logits[:, -1, :].float()
                if language_model_logits.shape != logits.shape:
                    raise ValueError("language-model logits must match decoder logits")
                if language_model_logits.device != logits.device:
                    raise ValueError("language-model and decoder logits must be on the same device")
                step_language_model_log_probs = torch.log_softmax(language_model_logits, dim=-1)

            step_non_ctc_log_probs = (
                torch.zeros_like(step_attention_log_probs)
                if ctc_weight == 1.0
                else (1.0 - ctc_weight) * step_attention_log_probs
            )
            if step_language_model_log_probs is not None:
                step_non_ctc_log_probs = step_non_ctc_log_probs + language_model_weight * step_language_model_log_probs
            candidate_non_ctc_log_probs = step_non_ctc_log_probs + alive_non_ctc_log_probs[:, None]

            candidate_ctc_states = None
            if ctc_prefix_scorer is None:
                candidate_log_probs = candidate_non_ctc_log_probs
            else:
                assert alive_ctc_states is not None
                candidate_ctc_log_probs, candidate_ctc_states = ctc_prefix_scorer.score_extensions(
                    last_token_ids[:, 0],
                    alive_ctc_states,
                    initial=position == 0,
                )
                candidate_log_probs = candidate_non_ctc_log_probs + ctc_weight * candidate_ctc_log_probs
            generated_length = position + 1
            penalty = self._length_penalty(generated_length, length_penalty)
            flat_candidate_scores = (candidate_log_probs / penalty).flatten()
            flat_indices = torch.arange(flat_candidate_scores.numel(), device=device)
            flat_token_ids = flat_indices.remainder(vocab_size)

            new_finished_scores, new_finished_indices = torch.topk(
                flat_candidate_scores.masked_fill(flat_token_ids.ne(self.eos_token_id), -torch.inf),
                beam_size,
            )
            new_finished_parents = torch.div(new_finished_indices, vocab_size, rounding_mode="floor")
            new_finished_token_ids = alive_token_ids.index_select(0, new_finished_parents).clone()
            new_finished_token_ids[:, position + 1] = self.eos_token_id
            combined_scores = torch.cat((finished_scores, new_finished_scores))
            combined_token_ids = torch.cat((finished_token_ids, new_finished_token_ids))
            new_lengths = torch.full((beam_size,), generated_length, dtype=torch.long, device=device)
            combined_lengths = torch.cat((finished_lengths, new_lengths))
            finished_scores, finished_indices = torch.topk(combined_scores, beam_size)
            finished_token_ids = combined_token_ids.index_select(0, finished_indices)
            finished_lengths = combined_lengths.index_select(0, finished_indices)

            _, alive_indices = torch.topk(
                flat_candidate_scores.masked_fill(flat_token_ids.eq(self.eos_token_id), -torch.inf),
                beam_size,
            )
            alive_parents = torch.div(alive_indices, vocab_size, rounding_mode="floor")
            alive_token_ids = alive_token_ids.index_select(0, alive_parents).clone()
            alive_token_ids[:, position + 1] = alive_indices.remainder(vocab_size)
            alive_log_probs = candidate_log_probs.flatten().index_select(0, alive_indices)
            alive_non_ctc_log_probs = candidate_non_ctc_log_probs.flatten().index_select(0, alive_indices)
            alive_caches = self._select_caches(step_caches, alive_parents)
            if step_language_model_cache is not None:
                alive_language_model_cache = self._select_language_model_cache(
                    step_language_model_cache,
                    alive_parents,
                )
            if candidate_ctc_states is not None:
                alive_ctc_states = candidate_ctc_states.flatten(0, 1).index_select(0, alive_indices)

            best_finished_score = finished_scores[0]
            best_alive_upper_bound = alive_log_probs.max() / self._length_penalty(max_new_tokens, length_penalty)
            if bool(torch.isfinite(best_finished_score) & (best_finished_score >= best_alive_upper_bound)):
                break

        if torch.isfinite(finished_scores[0]):
            best_length = int(finished_lengths[0].item())
            return EncoderDecoderBeamSearchResult(
                token_ids=finished_token_ids[0, : best_length + 1].tolist(),
                score=float(finished_scores[0].item()),
            )

        scores = alive_log_probs / self._length_penalty(generated_length, length_penalty)
        best_index = int(scores.argmax().item())
        return EncoderDecoderBeamSearchResult(
            token_ids=alive_token_ids[best_index, : generated_length + 1].tolist(),
            score=float(scores[best_index].item()),
        )

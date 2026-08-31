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


class EncoderDecoderBeamSearch:
    """Batched beam search for an autoregressive encoder-decoder model."""

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
        self.model = model
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.language_model = language_model

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
    ) -> EncoderDecoderBeamSearchResult:
        """Decode one encoded utterance, optionally using language-model shallow fusion.

        The returned sequence includes BOS and includes EOS when decoding completed. The score
        combines autoregressive decoder and weighted language-model log probabilities before
        applying the length penalty.
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
        if encoder_outputs.ndim != 3 or encoder_outputs.shape[0] != 1:
            raise ValueError("encoder_outputs must have shape (1, source_length, hidden_size)")
        expected_mask_shape = (1, encoder_outputs.shape[1])
        if encoder_attention_mask.shape != expected_mask_shape:
            raise ValueError(f"encoder_attention_mask must have shape {expected_mask_shape}")
        if encoder_attention_mask.device != encoder_outputs.device:
            raise ValueError("encoder outputs and attention mask must be on the same device")
        if not torch.any(encoder_attention_mask):
            raise ValueError("encoder_attention_mask must contain at least one valid frame")

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
        alive_caches: list[DecoderLayerCache] = []
        alive_language_model_cache: TransformerLMCache | None = None

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

            step_log_probs = torch.log_softmax(logits, dim=-1)
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
                step_log_probs = step_log_probs + language_model_weight * torch.log_softmax(
                    language_model_logits,
                    dim=-1,
                )

            candidate_log_probs = step_log_probs + alive_log_probs[:, None]
            generated_length = position + 1
            penalty = self._length_penalty(generated_length, length_penalty)
            flat_candidate_scores = (candidate_log_probs / penalty).flatten()
            num_candidates = min(2 * beam_size, flat_candidate_scores.numel())
            top_scores, top_indices = torch.topk(flat_candidate_scores, num_candidates)
            top_log_probs = candidate_log_probs.flatten().index_select(0, top_indices)
            parent_indices = torch.div(top_indices, vocab_size, rounding_mode="floor")
            next_token_ids = top_indices.remainder(vocab_size)

            candidate_token_ids = alive_token_ids.index_select(0, parent_indices).clone()
            candidate_token_ids[:, position + 1] = next_token_ids
            is_finished = next_token_ids.eq(self.eos_token_id)

            new_finished_scores = top_scores.masked_fill(~is_finished, -torch.inf)
            combined_scores = torch.cat((finished_scores, new_finished_scores))
            combined_token_ids = torch.cat((finished_token_ids, candidate_token_ids))
            new_lengths = torch.full((num_candidates,), generated_length, dtype=torch.long, device=device)
            combined_lengths = torch.cat((finished_lengths, new_lengths))
            finished_scores, finished_indices = torch.topk(combined_scores, beam_size)
            finished_token_ids = combined_token_ids.index_select(0, finished_indices)
            finished_lengths = combined_lengths.index_select(0, finished_indices)

            alive_scores = top_scores.masked_fill(is_finished, -torch.inf)
            _, alive_indices = torch.topk(alive_scores, beam_size)
            alive_token_ids = candidate_token_ids.index_select(0, alive_indices)
            alive_log_probs = top_log_probs.index_select(0, alive_indices)
            alive_parents = parent_indices.index_select(0, alive_indices)
            alive_caches = self._select_caches(step_caches, alive_parents)
            if step_language_model_cache is not None:
                alive_language_model_cache = self._select_language_model_cache(
                    step_language_model_cache,
                    alive_parents,
                )

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

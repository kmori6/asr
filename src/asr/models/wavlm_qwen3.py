import math
from typing import cast

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType
from transformers import Qwen3ForCausalLM, WavLMModel


class WavLMQwen3(nn.Module):
    """WavLM speech encoder connected to a Qwen3 causal language model.

    Args:
        speech_encoder (WavLMModel): WavLM model that converts padded waveforms into
            hidden states with shape ``(batch_size, num_frames, speech_hidden_size)``.
        language_model (Qwen3ForCausalLM): Qwen3 model that consumes projected audio
            embeddings followed by assistant-header and response token embeddings.
        audio_downsample_factor (int): Number of consecutive WavLM frames concatenated
            into one audio token. The default of four reduces WavLM's approximately
            50 Hz output to approximately 12.5 Hz.

    Notes:
        This model follows the encoder-projector-LLM structure proposed in the
        `Qwen3-ASR Technical Report <https://arxiv.org/abs/2601.21337>`_, but replaces
        AuT with WavLM. It does not reproduce AuT's dynamic attention windows. Stock
        WavLM uses full-context attention, so streaming applications must re-encode
        accumulated or overlapping audio and revise earlier hypotheses.
    """

    def __init__(
        self,
        speech_encoder: WavLMModel,
        language_model: Qwen3ForCausalLM,
        audio_downsample_factor: int = 4,
    ) -> None:
        super().__init__()
        if not isinstance(speech_encoder, WavLMModel):
            raise TypeError("speech_encoder must be an instance of WavLMModel")
        if not isinstance(language_model, Qwen3ForCausalLM):
            raise TypeError("language_model must be an instance of Qwen3ForCausalLM")
        if audio_downsample_factor <= 0:
            raise ValueError("audio_downsample_factor must be positive")
        if speech_encoder.config.add_adapter:
            raise ValueError("WavLM adapters are not supported")

        self.speech_encoder = speech_encoder
        self.language_model = language_model
        self.audio_downsample_factor = audio_downsample_factor
        stacked_size = speech_encoder.config.hidden_size * audio_downsample_factor
        self.audio_projector = nn.Sequential(
            nn.LayerNorm(stacked_size),
            nn.Linear(stacked_size, language_model.config.hidden_size),
        )
        token_embedding = cast(nn.Embedding, language_model.get_input_embeddings())
        embedding_weight = token_embedding.weight
        self.audio_projector.to(device=embedding_weight.device, dtype=embedding_weight.dtype)

    @classmethod
    def from_pretrained(
        cls,
        speech_encoder_name_or_path: str = "microsoft/wavlm-base-plus",
        language_model_name_or_path: str = "Qwen/Qwen3-0.6B",
        audio_downsample_factor: int = 4,
        dtype: torch.dtype | str | None = None,
    ) -> "WavLMQwen3":
        """Load pretrained WavLM and Qwen3 weights.

        Args:
            speech_encoder_name_or_path (str): Hugging Face model ID or local directory
                containing WavLM weights.
            language_model_name_or_path (str): Hugging Face model ID or local directory
                containing Qwen3 causal language-model weights.
            audio_downsample_factor (int): Number of WavLM frames concatenated into one
                projected audio token.
            dtype (torch.dtype | str | None): Weight dtype passed to both
                ``from_pretrained`` calls. ``None`` uses each checkpoint's default dtype.

        Returns:
            WavLMQwen3: Model initialized from both pretrained checkpoints.
        """
        if dtype is None:
            speech_encoder = WavLMModel.from_pretrained(speech_encoder_name_or_path)
            language_model = Qwen3ForCausalLM.from_pretrained(language_model_name_or_path)
        else:
            speech_encoder = WavLMModel.from_pretrained(speech_encoder_name_or_path, dtype=dtype)
            language_model = Qwen3ForCausalLM.from_pretrained(language_model_name_or_path, dtype=dtype)
        return cls(
            speech_encoder=speech_encoder,
            language_model=language_model,
            audio_downsample_factor=audio_downsample_factor,
        )

    @property
    def samples_per_audio_token(self) -> int:
        """Return the waveform stride represented by one projected audio token.

        Returns:
            int: Number of waveform samples between consecutive projected audio tokens.
        """
        return math.prod(self.speech_encoder.config.conv_stride) * self.audio_downsample_factor

    def forward(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute transcript-only causal language-model loss.

        Args:
            waveforms (torch.Tensor): Right-padded mono waveforms with shape
                ``(batch_size, num_samples)``.
            waveform_lengths (torch.Tensor): Valid waveform lengths in samples with shape
                ``(batch_size,)`` and dtype ``torch.long``.
            input_ids (torch.Tensor): Right-padded assistant-header and response token IDs
                with shape ``(batch_size, text_length)`` and dtype ``torch.long``.
            attention_mask (torch.Tensor | None): Valid text-token mask with shape
                ``(batch_size, text_length)``. ``None`` treats every text token as valid.
            labels (torch.Tensor | None): Causal targets with shape
                ``(batch_size, text_length)`` and dtype ``torch.long``. Prompt and padding
                positions must be ``-100`` so only response tokens contribute to the loss.

        Returns:
            dict[str, torch.Tensor]: Mapping containing ``logits`` with shape
                ``(batch_size, text_length, vocab_size)`` and, when ``labels`` is supplied,
                a scalar ``loss``.
        """
        self._validate_text_batch(waveforms, input_ids, attention_mask, labels)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        inputs_embeds, combined_attention_mask, audio_length = self._prepare_language_model_inputs(
            waveforms,
            waveform_lengths,
            input_ids,
            attention_mask,
        )
        combined_labels = None
        if labels is not None:
            audio_labels = labels.new_full((labels.shape[0], audio_length), -100)
            combined_labels = torch.cat((audio_labels, labels), dim=1)

        position_ids = combined_attention_mask.long().cumsum(dim=1) - 1
        position_ids.masked_fill_(~combined_attention_mask, 0)
        self._validate_context_length(inputs_embeds.shape[1])
        output = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_attention_mask,
            position_ids=position_ids,
            labels=combined_labels,
            use_cache=False,
        )
        result = {"logits": output.logits[:, audio_length:]}
        if output.loss is not None:
            result["loss"] = output.loss
        return result

    def encode_audio(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode speech into left-padded language-model audio tokens.

        Args:
            waveforms (torch.Tensor): Right-padded mono waveforms with shape
                ``(batch_size, num_samples)``.
            waveform_lengths (torch.Tensor): Valid waveform lengths in samples with shape
                ``(batch_size,)`` and dtype ``torch.long``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Projected audio embeddings with shape
                ``(batch_size, audio_length, language_hidden_size)`` and their boolean
                attention mask with shape ``(batch_size, audio_length)``. Valid audio tokens
                are aligned to the right so every sequence ends immediately before its text.
        """
        self._validate_audio_batch(waveforms, waveform_lengths)
        sample_positions = torch.arange(waveforms.shape[1], device=waveforms.device)
        waveform_attention_mask = sample_positions[None, :] < waveform_lengths[:, None]
        encoder_output = self.speech_encoder(
            input_values=waveforms,
            attention_mask=waveform_attention_mask.long(),
            return_dict=True,
        )

        feature_lengths = self._feature_lengths(waveform_lengths)
        feature_positions = torch.arange(encoder_output.last_hidden_state.shape[1], device=waveforms.device)
        feature_mask = feature_positions[None, :] < feature_lengths[:, None]
        audio_embeddings, audio_attention_mask = self._project_audio(
            encoder_output.last_hidden_state,
            feature_mask,
        )
        return self._left_pad(audio_embeddings, audio_attention_mask)

    @torch.inference_mode()
    def generate(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 128,
        num_beams: int = 1,
    ) -> torch.Tensor:
        """Generate transcript token IDs for a complete waveform.

        Args:
            waveforms (torch.Tensor): Right-padded mono waveforms with shape
                ``(batch_size, num_samples)``.
            waveform_lengths (torch.Tensor): Valid waveform lengths in samples with shape
                ``(batch_size,)`` and dtype ``torch.long``.
            input_ids (torch.Tensor): Left-padded assistant-header token IDs with shape
                ``(batch_size, text_length)`` and dtype ``torch.long``.
            attention_mask (torch.Tensor | None): Valid assistant-header mask with shape
                ``(batch_size, text_length)``. ``None`` treats every token as valid.
            max_new_tokens (int): Maximum number of transcript tokens to generate.
            num_beams (int): Beam width. One performs greedy decoding.

        Returns:
            torch.Tensor: Newly generated token IDs, excluding ``input_ids``, with shape
                ``(batch_size, generated_length)`` and dtype ``torch.long``. The generated
                length is at most ``max_new_tokens`` and may be shorter after EOS.

        Notes:
            Sampling is disabled to keep ASR decoding deterministic. The complete available
            waveform is encoded on every call; no WavLM or Qwen3 cache is retained across calls.
        """
        if self.training:
            raise RuntimeError("generation requires model.eval()")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if num_beams <= 0:
            raise ValueError("num_beams must be positive")
        self._validate_text_batch(waveforms, input_ids, attention_mask, labels=None)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if torch.any(~attention_mask[:, -1].bool()):
            raise ValueError("generation inputs must be left-padded, not right-padded")

        inputs_embeds, combined_attention_mask, _ = self._prepare_language_model_inputs(
            waveforms,
            waveform_lengths,
            input_ids,
            attention_mask,
        )
        self._validate_context_length(inputs_embeds.shape[1] + max_new_tokens)
        generated_ids = self.language_model.generate(  # type: ignore[misc]
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=combined_attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
        )
        if not isinstance(generated_ids, torch.Tensor):
            raise TypeError("language_model.generate must return token IDs")
        return generated_ids[:, input_ids.shape[1] :]

    def freeze_speech_encoder(self) -> None:
        """Freeze every WavLM parameter while leaving the projector and Qwen3 trainable."""
        self.speech_encoder.requires_grad_(False)

    def freeze_feature_encoder(self) -> None:
        """Freeze WavLM's waveform convolution while leaving its Transformer trainable."""
        self.speech_encoder.freeze_feature_encoder()

    def add_language_model_lora(
        self,
        rank: int,
        alpha: int,
        dropout: float,
        target_modules: list[str],
    ) -> None:
        """Add trainable LoRA adapters to Qwen3 and freeze its base weights.

        Args:
            rank (int): Rank of each low-rank update matrix.
            alpha (int): LoRA scaling factor.
            dropout (float): Dropout probability applied before LoRA updates.
            target_modules (list[str]): Qwen3 linear-module names that receive adapters.
        """
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if not target_modules:
            raise ValueError("LoRA target_modules must not be empty")
        self.language_model.add_adapter(  # type: ignore[no-untyped-call]
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=rank,
                lora_alpha=alpha,
                lora_dropout=dropout,
                bias="none",
                target_modules=target_modules,
            )
        )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict[str, object] | None = None) -> None:
        """Enable activation checkpointing in both pretrained Transformer stacks."""
        self.speech_encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        self.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self) -> None:
        """Disable activation checkpointing in both pretrained Transformer stacks."""
        self.speech_encoder.gradient_checkpointing_disable()
        self.language_model.gradient_checkpointing_disable()

    def _prepare_language_model_inputs(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Concatenate projected audio and text embeddings for Qwen3.

        Args:
            waveforms (torch.Tensor): Right-padded waveforms with shape
                ``(batch_size, num_samples)``.
            waveform_lengths (torch.Tensor): Valid waveform lengths with shape
                ``(batch_size,)``.
            input_ids (torch.Tensor): Text token IDs with shape
                ``(batch_size, text_length)``.
            attention_mask (torch.Tensor): Valid text-token mask with shape
                ``(batch_size, text_length)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor, int]: Combined embeddings with shape
                ``(batch_size, audio_length + text_length, language_hidden_size)``, their
                boolean attention mask with shape ``(batch_size, audio_length + text_length)``,
                and the padded ``audio_length``.
        """
        audio_embeddings, audio_attention_mask = self.encode_audio(waveforms, waveform_lengths)
        text_embeddings = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat((audio_embeddings, text_embeddings), dim=1)
        combined_attention_mask = torch.cat((audio_attention_mask, attention_mask.bool()), dim=1)
        return inputs_embeds, combined_attention_mask, audio_embeddings.shape[1]

    def _project_audio(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stack WavLM frames and project them to Qwen3's hidden size.

        Args:
            hidden_states (torch.Tensor): WavLM outputs with shape
                ``(batch_size, num_frames, speech_hidden_size)``.
            attention_mask (torch.Tensor): Boolean valid-frame mask with shape
                ``(batch_size, num_frames)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Right-padded projected embeddings with shape
                ``(batch_size, audio_length, language_hidden_size)`` and their boolean mask
                with shape ``(batch_size, audio_length)``, where
                ``audio_length = ceil(num_frames / audio_downsample_factor)``.
        """
        factor = self.audio_downsample_factor
        padding = (-hidden_states.shape[1]) % factor
        hidden_states = hidden_states.masked_fill(~attention_mask[:, :, None], 0.0)
        if padding:
            hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, padding))
            attention_mask = nn.functional.pad(attention_mask, (0, padding), value=False)

        batch_size, frame_count, hidden_size = hidden_states.shape
        stacked_states = hidden_states.reshape(batch_size, frame_count // factor, factor * hidden_size)
        stacked_mask = attention_mask.reshape(batch_size, frame_count // factor, factor).any(dim=-1)
        projection = cast(nn.Linear, self.audio_projector[1])
        stacked_states = stacked_states.to(dtype=projection.weight.dtype)
        embeddings = self.audio_projector(stacked_states)
        embeddings = embeddings.masked_fill(~stacked_mask[:, :, None], 0.0)
        return embeddings, stacked_mask

    @staticmethod
    def _left_pad(
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Move valid embeddings to the right without changing sequence length.

        Args:
            embeddings (torch.Tensor): Right-padded embeddings with shape
                ``(batch_size, sequence_length, hidden_size)``.
            attention_mask (torch.Tensor): Boolean right-padded mask with shape
                ``(batch_size, sequence_length)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Left-padded embeddings with the same shape and
                their boolean left-padded mask with shape ``(batch_size, sequence_length)``.
        """
        sequence_length = embeddings.shape[1]
        lengths = attention_mask.sum(dim=1)
        target_positions = torch.arange(sequence_length, device=embeddings.device)[None, :]
        source_positions = target_positions - (sequence_length - lengths[:, None])
        left_mask = source_positions >= 0
        gather_indices = source_positions.clamp_min(0)[:, :, None].expand(-1, -1, embeddings.shape[2])
        left_padded = embeddings.gather(dim=1, index=gather_indices)
        return left_padded.masked_fill(~left_mask[:, :, None], 0.0), left_mask

    def _feature_lengths(self, waveform_lengths: torch.Tensor) -> torch.Tensor:
        """Convert waveform lengths to WavLM feature lengths.

        Args:
            waveform_lengths (torch.Tensor): Waveform lengths in samples with shape
                ``(batch_size,)`` and dtype ``torch.long``.

        Returns:
            torch.Tensor: Feature lengths with shape ``(batch_size,)`` and dtype
                ``torch.long``.
        """
        feature_lengths = waveform_lengths
        for kernel_size, stride in zip(
            self.speech_encoder.config.conv_kernel,
            self.speech_encoder.config.conv_stride,
            strict=True,
        ):
            feature_lengths = torch.div(feature_lengths - kernel_size, stride, rounding_mode="floor") + 1
        return feature_lengths

    def _validate_audio_batch(self, waveforms: torch.Tensor, waveform_lengths: torch.Tensor) -> None:
        """Validate padded waveform tensors and their lengths.

        Args:
            waveforms (torch.Tensor): Padded waveforms with shape
                ``(batch_size, num_samples)``.
            waveform_lengths (torch.Tensor): Valid lengths with shape ``(batch_size,)``.
        """
        if waveforms.ndim != 2 or waveforms.shape[1] == 0:
            raise ValueError("waveforms must have shape (batch, num_samples) with at least one sample")
        if waveform_lengths.shape != (waveforms.shape[0],):
            raise ValueError(f"waveform_lengths must have shape ({waveforms.shape[0]},)")
        if waveform_lengths.dtype != torch.long:
            raise TypeError("waveform_lengths must have dtype torch.long")
        if waveform_lengths.device != waveforms.device:
            raise ValueError("waveforms and waveform_lengths must be on the same device")
        if torch.any(waveform_lengths <= 0) or torch.any(waveform_lengths > waveforms.shape[1]):
            raise ValueError("waveform lengths must be positive and fit within waveforms")
        if torch.any(self._feature_lengths(waveform_lengths) <= 0):
            raise ValueError("waveforms are too short for the WavLM feature encoder")

    def _validate_context_length(self, sequence_length: int) -> None:
        """Validate an audio-text sequence length against Qwen3's context limit.

        Args:
            sequence_length (int): Total number of projected audio, text, and reserved
                generation positions.
        """
        if sequence_length > self.language_model.config.max_position_embeddings:
            raise ValueError(
                f"audio and text require {sequence_length} positions, but Qwen3 supports "
                f"{self.language_model.config.max_position_embeddings}"
            )

    def _validate_text_batch(
        self,
        waveforms: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        labels: torch.Tensor | None,
    ) -> None:
        """Validate text tensors consumed by training or generation.

        Args:
            waveforms (torch.Tensor): Padded waveforms with shape
                ``(batch_size, num_samples)``; only the batch dimension is inspected.
            input_ids (torch.Tensor): Text token IDs with shape
                ``(batch_size, text_length)`` and dtype ``torch.long``.
            attention_mask (torch.Tensor | None): Valid text-token mask with shape
                ``(batch_size, text_length)``.
            labels (torch.Tensor | None): Causal targets with shape
                ``(batch_size, text_length)``. Ignored positions contain ``-100``.
        """
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have shape (batch, text_length) with at least one token")
        if input_ids.shape[0] != waveforms.shape[0]:
            raise ValueError("waveforms and input_ids must have the same batch size")
        if input_ids.dtype != torch.long:
            raise TypeError("input_ids must have dtype torch.long")
        if input_ids.device != waveforms.device:
            raise ValueError("waveforms and input_ids must be on the same device")
        if torch.any(input_ids < 0) or torch.any(input_ids >= self.language_model.config.vocab_size):
            raise ValueError("input_ids must contain valid Qwen3 vocabulary indices")
        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must have the same shape as input_ids")
            if attention_mask.device != waveforms.device:
                raise ValueError("attention_mask must be on the same device as waveforms")
            if torch.any(attention_mask.sum(dim=1) == 0):
                raise ValueError("each text sequence must contain at least one valid token")
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            if labels.dtype != torch.long or labels.device != waveforms.device:
                raise ValueError("labels must be torch.long and on the same device as waveforms")
            valid_labels = labels.ne(-100)
            if not torch.any(valid_labels):
                raise ValueError("labels must contain at least one supervised response token")
            if attention_mask is not None:
                if torch.any(~attention_mask[:, 0].bool()):
                    raise ValueError("training text must be right-padded so it follows the audio prefix")
                if torch.any(valid_labels & ~attention_mask.bool()):
                    raise ValueError("labels at padded text positions must be -100")
            if torch.any(labels[valid_labels] < 0) or torch.any(
                labels[valid_labels] >= self.language_model.config.vocab_size
            ):
                raise ValueError("labels must contain Qwen3 vocabulary indices or -100")

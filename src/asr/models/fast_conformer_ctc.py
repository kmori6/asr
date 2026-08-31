from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn

from asr.modules.conformer import FastConformer, FastConformerCache, StreamingFastConformer
from asr.modules.frontend import LogMelSpectrogram, SpecAugment


class FastConformerCTC(nn.Module):
    """Full-context FastConformer with a CTC output layer."""

    _encoder_type: type[FastConformer] = FastConformer

    def __init__(
        self,
        frontend: LogMelSpectrogram,
        spec_augment: SpecAugment,
        encoder: FastConformer,
        vocab_size: int,
        blank_token_id: int,
    ) -> None:
        super().__init__()
        if not isinstance(encoder, self._encoder_type):
            raise TypeError(f"encoder must be an instance of {self._encoder_type.__name__}")
        if encoder.input_size != frontend.n_mels:
            raise ValueError("encoder input_size must match frontend n_mels")
        if vocab_size <= 1:
            raise ValueError("vocab_size must contain a blank and at least one label")
        if not 0 <= blank_token_id < vocab_size:
            raise ValueError("blank_token_id must be a valid vocabulary index")

        self.frontend = frontend
        self.spec_augment = spec_augment
        self.encoder = encoder
        self.vocab_size = vocab_size
        self.blank_token_id = blank_token_id
        self.output_projection = nn.Linear(encoder.hidden_size, vocab_size)
        self.ctc_loss_fn = nn.CTCLoss(blank=blank_token_id, reduction="sum", zero_infinity=True)

    def forward(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor,
        labels: torch.Tensor,
        label_lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute the mean CTC loss for a padded waveform batch."""
        return self._forward(waveforms, waveform_lengths, labels, label_lengths, chunk_size=None)

    def _forward(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor,
        labels: torch.Tensor,
        label_lengths: torch.Tensor,
        chunk_size: int | None,
    ) -> dict[str, torch.Tensor]:
        self._validate_batch(waveforms, waveform_lengths, labels, label_lengths)
        encoder_outputs, encoder_lengths = self._encode(waveforms, waveform_lengths, chunk_size)
        log_probs = self.logits(encoder_outputs).float().log_softmax(dim=-1).transpose(0, 1)
        loss = self.ctc_loss_fn(log_probs, labels, encoder_lengths, label_lengths) / waveforms.shape[0]
        return {"loss": loss}

    def encode(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode padded waveforms with full-context attention."""
        return self._encode(waveforms, waveform_lengths, chunk_size=None)

    def _encode(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor,
        chunk_size: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if chunk_size is not None:
            raise ValueError("full-context FastConformer does not accept chunk_size")
        features, feature_lengths = self.frontend(waveforms, waveform_lengths)
        features = self.spec_augment(features, feature_lengths)
        return self.encoder(features, feature_lengths)

    def logits(self, encoder_outputs: torch.Tensor) -> torch.Tensor:
        """Project encoder outputs to CTC vocabulary logits."""
        return self.output_projection(encoder_outputs)

    def _validate_batch(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor,
        labels: torch.Tensor,
        label_lengths: torch.Tensor,
    ) -> None:
        if waveforms.ndim != 2 or waveforms.shape[1] == 0:
            raise ValueError("waveforms must have shape (batch, num_samples) with at least one sample")
        batch_size = waveforms.shape[0]
        if waveform_lengths.shape != (batch_size,):
            raise ValueError(f"waveform_lengths must have shape ({batch_size},)")
        if labels.ndim != 2 or labels.shape[0] != batch_size:
            raise ValueError("labels must have shape (batch, max_label_length)")
        if label_lengths.shape != (batch_size,):
            raise ValueError(f"label_lengths must have shape ({batch_size},)")
        if waveform_lengths.dtype != torch.long or labels.dtype != torch.long or label_lengths.dtype != torch.long:
            raise TypeError("waveform_lengths, labels, and label_lengths must have dtype torch.long")
        if any(tensor.device != waveforms.device for tensor in (waveform_lengths, labels, label_lengths)):
            raise ValueError("all batch tensors must be on the same device")
        if torch.any(waveform_lengths <= 0) or torch.any(waveform_lengths > waveforms.shape[1]):
            raise ValueError("waveform lengths must be positive and fit within waveforms")
        if torch.any(label_lengths <= 0) or torch.any(label_lengths > labels.shape[1]):
            raise ValueError("label lengths must be positive and fit within labels")
        if torch.any(labels < 0) or torch.any(labels >= self.vocab_size):
            raise ValueError("labels must contain valid vocabulary indices")
        positions = torch.arange(labels.shape[1], device=labels.device)
        valid_labels = positions[None, :] < label_lengths[:, None]
        if torch.any(labels[valid_labels] == self.blank_token_id):
            raise ValueError("valid labels must not contain the blank token")


@dataclass(frozen=True, slots=True)
class StreamingFastConformerCTCCache:
    """Frontend and encoder states retained across waveform chunks."""

    frontend: torch.Tensor | None
    encoder: FastConformerCache | None


class StreamingFastConformerCTC(FastConformerCTC):
    """FastConformer CTC trained and evaluated with chunkwise streaming."""

    _encoder_type = StreamingFastConformer

    def forward(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor,
        labels: torch.Tensor,
        label_lengths: torch.Tensor,
        chunk_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute CTC loss using a sampled or explicit encoder chunk size."""
        return self._forward(waveforms, waveform_lengths, labels, label_lengths, chunk_size)

    def encode(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor,
        chunk_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode padded waveforms with chunk-aware attention."""
        return self._encode(waveforms, waveform_lengths, chunk_size)

    def _encode(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor,
        chunk_size: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features, feature_lengths = self.frontend(waveforms, waveform_lengths)
        features = self.spec_augment(features, feature_lengths)
        encoder = cast(StreamingFastConformer, self.encoder)
        return encoder(features, feature_lengths, chunk_size)

    def encode_chunk(
        self,
        waveforms: torch.Tensor,
        cache: StreamingFastConformerCTCCache | None = None,
        chunk_size: int | None = None,
        is_final: bool = False,
    ) -> tuple[torch.Tensor, StreamingFastConformerCTCCache]:
        """Encode the next waveform chunk and update frontend and encoder caches."""
        frontend_cache = None if cache is None else cache.frontend
        encoder_cache = None if cache is None else cache.encoder
        features, next_frontend_cache = self.frontend.forward_chunk(waveforms, frontend_cache)
        feature_lengths = torch.tensor([features.shape[1]], dtype=torch.long, device=features.device)
        features = self.spec_augment(features, feature_lengths)
        encoder = cast(StreamingFastConformer, self.encoder)
        encoder_outputs, next_encoder_cache = encoder.forward_chunk(
            features,
            cache=encoder_cache,
            chunk_size=chunk_size,
            is_final=is_final,
        )
        return encoder_outputs, StreamingFastConformerCTCCache(
            frontend=next_frontend_cache,
            encoder=next_encoder_cache,
        )

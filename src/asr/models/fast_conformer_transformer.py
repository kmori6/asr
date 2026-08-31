import math

import torch
import torch.nn as nn

from asr.modules.conformer import FastConformer
from asr.modules.frontend import LogMelSpectrogram, SpecAugment
from asr.modules.transformer.cache import DecoderLayerCache
from asr.modules.transformer.decoder import Decoder
from asr.modules.transformer.decoder_layer import DecoderLayer
from asr.modules.transformer.feed_forward import FeedForward
from asr.modules.transformer.multi_head_attention import MultiHeadAttention
from asr.modules.transformer.positional_encoding import PositionalEncoding


class FastConformerTransformer(nn.Module):
    """FastConformer encoder with an autoregressive Transformer decoder."""

    def __init__(
        self,
        frontend: LogMelSpectrogram,
        spec_augment: SpecAugment,
        encoder: FastConformer,
        vocab_size: int,
        blank_token_id: int,
        ctc_loss_weight: float = 0.2,
        decoder_hidden_size: int = 512,
        decoder_num_layers: int = 6,
        decoder_num_heads: int = 8,
        decoder_feed_forward_size: int = 2048,
        decoder_dropout_rate: float = 0.1,
        decoder_max_length: int = 512,
        label_smoothing: float = 0.1,
        ignore_index: int = -100,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(encoder, FastConformer):
            raise TypeError("encoder must be an instance of FastConformer")
        if encoder.input_size != frontend.n_mels:
            raise ValueError("encoder input_size must match frontend n_mels")
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if not 0 <= blank_token_id < vocab_size:
            raise ValueError("blank_token_id must be a valid vocabulary index")
        if not 0.0 <= ctc_loss_weight <= 1.0:
            raise ValueError("ctc_loss_weight must be in [0, 1]")
        if decoder_hidden_size != encoder.hidden_size:
            raise ValueError("decoder_hidden_size must match encoder hidden_size")
        if decoder_num_layers <= 0:
            raise ValueError("decoder_num_layers must be positive")
        if decoder_max_length <= 0:
            raise ValueError("decoder_max_length must be positive")

        self.frontend = frontend
        self.spec_augment = spec_augment
        self.encoder = encoder
        self.vocab_size = vocab_size
        self.blank_token_id = blank_token_id
        self.ctc_loss_weight = ctc_loss_weight
        self.decoder_hidden_size = decoder_hidden_size
        self.ignore_index = ignore_index

        self.embedding = nn.Embedding(vocab_size, decoder_hidden_size)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=decoder_hidden_size**-0.5)
        self.position = PositionalEncoding(decoder_hidden_size, decoder_max_length)
        self.dropout = nn.Dropout(decoder_dropout_rate)
        self.decoder = Decoder(
            layers=[
                DecoderLayer(
                    self_mha=MultiHeadAttention(
                        decoder_hidden_size,
                        decoder_num_heads,
                        decoder_dropout_rate,
                        bias=bias,
                    ),
                    self_mha_norm=nn.LayerNorm(decoder_hidden_size),
                    cross_mha=MultiHeadAttention(
                        decoder_hidden_size,
                        decoder_num_heads,
                        decoder_dropout_rate,
                        bias=bias,
                    ),
                    cross_mha_norm=nn.LayerNorm(decoder_hidden_size),
                    ffn=FeedForward(
                        decoder_hidden_size,
                        decoder_feed_forward_size,
                        decoder_dropout_rate,
                        activation=nn.ReLU(),
                        bias=bias,
                    ),
                    ffn_norm=nn.LayerNorm(decoder_hidden_size),
                    dropout_rate=decoder_dropout_rate,
                )
                for _ in range(decoder_num_layers)
            ],
            final_norm=nn.LayerNorm(decoder_hidden_size),
        )
        self.output_projection = nn.Linear(decoder_hidden_size, vocab_size, bias=False)
        self.output_projection.weight = self.embedding.weight
        self.ctc_projection = nn.Linear(encoder.hidden_size, vocab_size)
        self.ctc_loss_fn = nn.CTCLoss(blank=blank_token_id, reduction="sum", zero_infinity=True)
        self.cross_entropy_loss_fn = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction="sum",
        )

    def forward(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor,
        ctc_labels: torch.Tensor,
        ctc_label_lengths: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute the joint CTC and label-smoothed cross-entropy objective.

        Args:
            waveforms: Padded waveforms with shape ``(batch, num_samples)``.
            waveform_lengths: Valid waveform lengths with shape ``(batch,)``.
            ctc_labels: Padded transcript IDs without BOS or EOS, with shape
                ``(batch, ctc_target_length)``.
            ctc_label_lengths: Valid CTC target lengths with shape ``(batch,)``.
            decoder_input_ids: Right-shifted target IDs with shape ``(batch, target_length)``.
            decoder_attention_mask: Valid decoder-token mask with the same shape.
            labels: Following-token targets with the same shape. Padding must be ``ignore_index``.

        Returns:
            Scalar total loss, component losses, and decoder token accuracy.
        """
        self._validate_batch(
            waveforms,
            waveform_lengths,
            ctc_labels,
            ctc_label_lengths,
            decoder_input_ids,
            decoder_attention_mask,
            labels,
        )
        encoder_outputs, encoder_attention_mask = self.encode(waveforms, waveform_lengths)
        encoder_lengths = encoder_attention_mask.sum(dim=1)

        target_length = decoder_input_ids.shape[1]
        cross_attention_mask = encoder_attention_mask[:, None, :].expand(-1, target_length, -1)
        causal_mask = torch.ones(target_length, target_length, dtype=torch.bool, device=waveforms.device).tril()
        self_attention_mask = decoder_attention_mask[:, None, :].bool() & causal_mask[None, :, :]

        decoder_inputs = self.dropout(self.embed(decoder_input_ids))
        decoder_outputs = self.decoder(
            encoder_outputs,
            decoder_inputs,
            cross_attention_mask,
            self_attention_mask,
        )
        logits = self.logits(decoder_outputs)
        batch_size = waveforms.shape[0]
        ctc_log_probs = self.ctc_log_probs(encoder_outputs).transpose(0, 1)
        ctc_loss = self.ctc_loss_fn(ctc_log_probs, ctc_labels, encoder_lengths, ctc_label_lengths) / batch_size
        cross_entropy_loss = self.cross_entropy_loss_fn(logits.flatten(0, 1), labels.flatten()) / batch_size
        loss = self.ctc_loss_weight * ctc_loss + (1.0 - self.ctc_loss_weight) * cross_entropy_loss
        valid_labels = labels.ne(self.ignore_index)
        accuracy = logits.argmax(dim=-1)[valid_labels].eq(labels[valid_labels]).float().mean()
        return {
            "loss": loss,
            "ctc_loss": ctc_loss,
            "cross_entropy_loss": cross_entropy_loss,
            "accuracy": accuracy,
        }

    def encode(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode padded waveforms and return encoder outputs and their valid-frame mask."""
        features, feature_lengths = self.frontend(waveforms, waveform_lengths)
        features = self.spec_augment(features, feature_lengths)
        encoder_outputs, encoder_lengths = self.encoder(features, feature_lengths)
        positions = torch.arange(encoder_outputs.shape[1], device=encoder_outputs.device)
        attention_mask = positions[None, :] < encoder_lengths[:, None]
        return encoder_outputs, attention_mask

    def embed(self, token_ids: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Embed decoder token IDs and add sinusoidal positions."""
        embeddings = self.embedding(token_ids) * math.sqrt(self.decoder_hidden_size)
        return embeddings + self.position(embeddings, offset=offset)

    def logits(self, decoder_outputs: torch.Tensor) -> torch.Tensor:
        """Project decoder outputs to vocabulary logits."""
        return self.output_projection(decoder_outputs)

    def ctc_log_probs(self, encoder_outputs: torch.Tensor) -> torch.Tensor:
        """Return float32 CTC log probabilities shaped ``(batch, frames, vocab)``."""
        return self.ctc_projection(encoder_outputs).float().log_softmax(dim=-1)

    @torch.inference_mode()
    def predict(
        self,
        encoder_outputs: torch.Tensor,
        decoder_inputs: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        caches: list[DecoderLayerCache],
    ) -> tuple[torch.Tensor, list[DecoderLayerCache]]:
        """Decode one embedded token and update the decoder KV caches."""
        return self.decoder.predict(
            encoder_outputs,
            decoder_inputs,
            encoder_attention_mask,
            caches,
        )

    def _validate_batch(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor,
        ctc_labels: torch.Tensor,
        ctc_label_lengths: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        batch_size = waveforms.shape[0]
        if decoder_input_ids.ndim != 2 or decoder_input_ids.shape[1] == 0:
            raise ValueError("decoder_input_ids must have shape (batch, target_length) with at least one token")
        if decoder_input_ids.dtype != torch.long:
            raise TypeError("decoder_input_ids must have dtype torch.long")
        if decoder_input_ids.shape[0] != batch_size:
            raise ValueError("waveforms and decoder_input_ids must have the same batch size")
        if ctc_labels.ndim != 2 or ctc_labels.shape[0] != batch_size:
            raise ValueError("ctc_labels must have shape (batch, ctc_target_length)")
        if waveform_lengths.shape != (batch_size,):
            raise ValueError(f"waveform_lengths must have shape ({batch_size},)")
        if ctc_label_lengths.shape != (batch_size,):
            raise ValueError(f"ctc_label_lengths must have shape ({batch_size},)")
        if decoder_attention_mask.shape != decoder_input_ids.shape or labels.shape != decoder_input_ids.shape:
            raise ValueError("decoder_attention_mask and labels must match decoder_input_ids")
        integer_tensors = (waveform_lengths, ctc_labels, ctc_label_lengths, labels)
        if any(tensor.dtype != torch.long for tensor in integer_tensors):
            raise TypeError("waveform lengths, CTC targets, and decoder labels must have dtype torch.long")
        tensors = (waveform_lengths, ctc_labels, ctc_label_lengths, decoder_input_ids, decoder_attention_mask, labels)
        if any(tensor.device != waveforms.device for tensor in tensors):
            raise ValueError("all batch tensors must be on the same device")
        if torch.any(ctc_label_lengths <= 0) or torch.any(ctc_label_lengths > ctc_labels.shape[1]):
            raise ValueError("CTC label lengths must be positive and fit within ctc_labels")
        if torch.any(ctc_labels < 0) or torch.any(ctc_labels >= self.vocab_size):
            raise ValueError("ctc_labels must contain valid vocabulary indices")
        ctc_positions = torch.arange(ctc_labels.shape[1], device=ctc_labels.device)
        valid_ctc_labels = ctc_positions[None, :] < ctc_label_lengths[:, None]
        if torch.any(ctc_labels[valid_ctc_labels] == self.blank_token_id):
            raise ValueError("valid CTC labels must not contain the blank token")
        if torch.any(decoder_input_ids < 0) or torch.any(decoder_input_ids >= self.vocab_size):
            raise ValueError("decoder_input_ids must contain valid vocabulary indices")
        valid_labels = labels.ne(self.ignore_index)
        if not torch.any(valid_labels):
            raise ValueError("labels must contain at least one non-padding target")
        if torch.any(labels[valid_labels] < 0) or torch.any(labels[valid_labels] >= self.vocab_size):
            raise ValueError("non-padding labels must contain valid vocabulary indices")

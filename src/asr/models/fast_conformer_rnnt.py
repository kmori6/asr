from dataclasses import dataclass

import torch
import torch.nn as nn
from warprnnt_numba import RNNTLossNumba

from asr.modules.conformer import FastConformerEncoder, FastConformerEncoderCache
from asr.modules.frontend import LogMelSpectrogram, LogMelSpectrogramCache, SpecAugment
from asr.modules.rnnt import JointNetwork, PredictionNetwork


@dataclass(frozen=True, slots=True)
class FastConformerRNNTCache:
    """Frontend and encoder states retained across waveform chunks."""

    frontend: LogMelSpectrogramCache | None
    encoder: FastConformerEncoderCache | None


class FastConformerRNNT(nn.Module):
    """FastConformer RNN-T trained with RNN-T and auxiliary CTC losses."""

    def __init__(
        self,
        frontend: LogMelSpectrogram,
        spec_augment: SpecAugment,
        encoder: FastConformerEncoder,
        prediction_network: PredictionNetwork,
        joint_network: JointNetwork,
        ctc_loss_weight: float,
        fastemit_lambda: float,
    ) -> None:
        super().__init__()
        if encoder.input_size != frontend.n_mels:
            raise ValueError("encoder input_size must match the frontend n_mels")
        if joint_network.encoder_size != encoder.hidden_size:
            raise ValueError("joint encoder_size must match the encoder hidden_size")
        if joint_network.predictor_size != prediction_network.hidden_size:
            raise ValueError("joint predictor_size must match the prediction network hidden_size")
        if joint_network.vocab_size != prediction_network.vocab_size:
            raise ValueError("joint and prediction network vocabulary sizes must match")
        if ctc_loss_weight < 0.0:
            raise ValueError("ctc_loss_weight must be non-negative")
        if fastemit_lambda < 0.0:
            raise ValueError("fastemit_lambda must be non-negative")

        self.frontend = frontend
        self.spec_augment = spec_augment
        self.encoder = encoder
        self.prediction_network = prediction_network
        self.joint_network = joint_network
        self.blank_token_id = prediction_network.blank_token_id
        self.vocab_size = prediction_network.vocab_size
        self.ctc_loss_weight = ctc_loss_weight
        self.fastemit_lambda = fastemit_lambda

        self.rnnt_loss_fn = RNNTLossNumba(
            blank=self.blank_token_id,
            reduction="sum",
            fastemit_lambda=fastemit_lambda,
        )
        self.ctc_projection = nn.Linear(encoder.hidden_size, self.vocab_size)
        self.ctc_loss_fn = nn.CTCLoss(blank=self.blank_token_id, reduction="sum", zero_infinity=True)

    def forward(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor,
        labels: torch.Tensor,
        label_lengths: torch.Tensor,
        chunk_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            waveforms: Padded waveforms with shape ``(batch, num_samples)``.
            waveform_lengths: Valid waveform lengths with shape ``(batch,)``.
            labels: Padded target token IDs with shape
                ``(batch, max_label_length)``.
            label_lengths: Valid target lengths with shape ``(batch,)``.
            chunk_size: Optional attention chunk size measured after 8x
                subsampling. See ``FastConformerEncoder.forward``.

        Returns:
            Scalar total, RNN-T, and CTC losses.
        """
        self._validate_batch(waveforms, waveform_lengths, labels, label_lengths)
        encoder_outputs, encoder_lengths = self.encode(waveforms, waveform_lengths, chunk_size)

        blank_tokens = labels.new_full((labels.shape[0], 1), self.blank_token_id)
        predictor_inputs = torch.cat((blank_tokens, labels), dim=1)
        predictor_outputs, _ = self.prediction_network(predictor_inputs)

        logits = self.joint_network(encoder_outputs, predictor_outputs)
        rnnt_loss = self._rnnt_loss(logits, encoder_lengths, labels, label_lengths)
        ctc_loss = self._ctc_loss(encoder_outputs, encoder_lengths, labels, label_lengths)
        loss = rnnt_loss + self.ctc_loss_weight * ctc_loss
        return {
            "loss": loss,
            "rnnt_loss": rnnt_loss,
            "ctc_loss": ctc_loss,
        }

    def encode(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor,
        chunk_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            waveforms: Padded waveforms with shape ``(batch, num_samples)``.
            waveform_lengths: Valid waveform lengths with shape ``(batch,)``.
            chunk_size: Optional attention chunk size measured after 8x
                subsampling.

        Returns:
            Encoder representations and their valid lengths.
        """
        features, feature_lengths = self.frontend(waveforms, waveform_lengths)
        features = self.spec_augment(features, feature_lengths)
        return self.encoder(features, feature_lengths, chunk_size)

    def encode_chunk(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor,
        cache: FastConformerRNNTCache | None = None,
        chunk_size: int | None = None,
        is_final: bool = False,
    ) -> tuple[torch.Tensor, FastConformerRNNTCache]:
        """
        Args:
            waveforms: Next padded waveform chunk for one utterance with shape
                ``(1, num_chunk_samples)``.
            waveform_lengths: Valid samples in the chunk with shape ``(1,)``.
            cache: Frontend and encoder states from the preceding chunk, or
                ``None`` for the first chunk.
            chunk_size: Attention chunk size after 8x subsampling.
            is_final: If ``True``, flush the encoder's final incomplete chunk.

        Returns:
            Newly available encoder representations and the updated model cache.
        """
        if waveforms.ndim != 2 or waveforms.shape[0] != 1:
            raise ValueError(
                f"streaming waveforms must have shape (1, num_chunk_samples), but got {tuple(waveforms.shape)}"
            )
        if waveform_lengths.shape != (1,):
            raise ValueError("streaming waveform_lengths must have shape (1,)")

        frontend_cache = None if cache is None else cache.frontend
        encoder_cache = None if cache is None else cache.encoder
        features, feature_lengths, next_frontend_cache = self.frontend.forward_chunk(
            waveforms,
            waveform_lengths,
            frontend_cache,
        )
        features = self.spec_augment(features, feature_lengths)
        num_valid_frames = int(feature_lengths[0].item())
        encoder_outputs, next_encoder_cache = self.encoder.forward_chunk(
            features[:, :num_valid_frames],
            cache=encoder_cache,
            chunk_size=chunk_size,
            is_final=is_final,
        )
        next_cache = FastConformerRNNTCache(
            frontend=next_frontend_cache,
            encoder=next_encoder_cache,
        )
        return encoder_outputs, next_cache

    def _rnnt_loss(
        self,
        logits: torch.Tensor,
        encoder_lengths: torch.Tensor,
        labels: torch.Tensor,
        label_lengths: torch.Tensor,
    ) -> torch.Tensor:
        loss = self.rnnt_loss_fn(
            logits.float(),
            labels.int(),
            encoder_lengths.int(),
            label_lengths.int(),
        )
        return loss.sum() / labels.shape[0]

    def _ctc_loss(
        self,
        encoder_outputs: torch.Tensor,
        encoder_lengths: torch.Tensor,
        labels: torch.Tensor,
        label_lengths: torch.Tensor,
    ) -> torch.Tensor:
        log_probs = self.ctc_projection(encoder_outputs).float().log_softmax(dim=-1).transpose(0, 1)
        loss = self.ctc_loss_fn(log_probs, labels, encoder_lengths, label_lengths)
        return loss / labels.shape[0]

    def _validate_batch(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor,
        labels: torch.Tensor,
        label_lengths: torch.Tensor,
    ) -> None:
        if labels.ndim != 2:
            raise ValueError(f"labels must have shape (batch, max_label_length), but got {tuple(labels.shape)}")
        if labels.dtype != torch.long:
            raise ValueError("labels must have dtype torch.long")
        if labels.shape[0] != waveforms.shape[0]:
            raise ValueError("waveforms and labels must have the same batch size")
        if waveform_lengths.shape != (waveforms.shape[0],):
            raise ValueError(f"waveform_lengths must have shape ({waveforms.shape[0]},)")
        if label_lengths.shape != (labels.shape[0],):
            raise ValueError(f"label_lengths must have shape ({labels.shape[0]},)")
        if waveform_lengths.dtype != torch.long or label_lengths.dtype != torch.long:
            raise ValueError("waveform_lengths and label_lengths must have dtype torch.long")
        if (
            waveform_lengths.device != waveforms.device
            or labels.device != waveforms.device
            or label_lengths.device != waveforms.device
        ):
            raise ValueError("all batch tensors must be on the same device")
        if torch.any(label_lengths <= 0) or torch.any(label_lengths > labels.shape[1]):
            raise ValueError("label lengths must be positive and no greater than the padded label length")
        if torch.any(labels < 0) or torch.any(labels >= self.vocab_size):
            raise ValueError("labels must contain valid vocabulary indices")
        positions = torch.arange(labels.shape[1], device=labels.device)
        valid_labels = positions[None, :] < label_lengths[:, None]
        if torch.any(labels[valid_labels] == self.blank_token_id):
            raise ValueError("valid labels must not contain the blank token")

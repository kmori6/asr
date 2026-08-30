import torch
import torch.nn as nn

from asr.modules.transformer.cache import DecoderLayerCache
from asr.modules.transformer.multi_head_attention import MultiHeadAttention


class DecoderLayer(nn.Module):
    """Pre-normalized Transformer decoder layer."""

    def __init__(
        self,
        self_mha: MultiHeadAttention,
        self_mha_norm: nn.Module,
        cross_mha: MultiHeadAttention,
        cross_mha_norm: nn.Module,
        ffn: nn.Module,
        ffn_norm: nn.Module,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.self_mha = self_mha
        self.self_mha_norm = self_mha_norm
        self.self_mha_dropout = nn.Dropout(dropout_rate)
        self.cross_mha = cross_mha
        self.cross_mha_norm = cross_mha_norm
        self.cross_mha_dropout = nn.Dropout(dropout_rate)
        self.ffn = ffn
        self.ffn_norm = ffn_norm
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        x_enc: torch.Tensor,
        x_dec: torch.Tensor,
        mask_enc: torch.Tensor,
        mask_dec: torch.Tensor,
    ) -> torch.Tensor:
        """

        Args:
            x_enc (torch.Tensor): Encoder embedding sequence tensor
                (batch_size, source_sequence_length, hidden_size).
            x_dec (torch.Tensor): Decoder embedding sequence tensor
                (batch_size, target_sequence_length, hidden_size).
            mask_enc (torch.Tensor): Encoder-decoder attention mask
                (batch_size, target_sequence_length, source_sequence_length).
            mask_dec (torch.Tensor): Decoder self-attention mask
                (batch_size, target_sequence_length, target_sequence_length).

        Returns:
            torch.Tensor: Output sequence tensor
                (batch_size, target_sequence_length, hidden_size).
        """
        residual = x_dec
        x_dec = self.self_mha_norm(x_dec)
        x_dec = residual + self.self_mha_dropout(self.self_mha(x_dec, x_dec, x_dec, mask_dec))

        residual = x_dec
        x_dec = self.cross_mha_norm(x_dec)
        x_dec = residual + self.cross_mha_dropout(self.cross_mha(x_dec, x_enc, x_enc, mask_enc))

        residual = x_dec
        x_dec = self.ffn_norm(x_dec)
        x_dec = residual + self.ffn_dropout(self.ffn(x_dec))
        return x_dec

    @torch.inference_mode()
    def predict(
        self,
        x_enc: torch.Tensor,
        x_dec: torch.Tensor,
        mask_enc: torch.Tensor,
        mask_dec: torch.Tensor,
        cache: DecoderLayerCache | None = None,
    ) -> tuple[torch.Tensor, DecoderLayerCache]:
        """

        Args:
            x_enc (torch.Tensor): Encoder embedding sequence tensor
                (batch_size, source_sequence_length, hidden_size).
            x_dec (torch.Tensor): New decoder embedding sequence tensor
                (batch_size, target_sequence_length, hidden_size). In the current autoregressive
                decoding path, target_sequence_length is 1 when ``cache`` is provided.
            mask_enc (torch.Tensor): Encoder-decoder attention mask
                (batch_size, target_sequence_length, source_sequence_length).
            mask_dec (torch.Tensor): Decoder self-attention mask
                (batch_size, target_sequence_length, cached_length + target_sequence_length).
            cache (DecoderLayerCache | None): Projected self- and cross-attention keys and values
                from previous steps, or ``None`` on the first step.

        Returns:
            tuple[torch.Tensor, DecoderLayerCache]: Output sequence tensor and updated attention
                caches.

        Note:
            A cache does not inherently require a single-token input. Chunked decoding can use
            target_sequence_length greater than 1 when ``mask_dec`` provides the corresponding
            causal mask.
        """
        self_attention_cache = cache.self_attention if cache is not None else None
        cross_attention_cache = cache.cross_attention if cache is not None else None

        residual = x_dec
        x_dec = self.self_mha_norm(x_dec)
        self_mha_output, self_attention_cache = self.self_mha.forward_with_cache(
            x_dec,
            x_dec,
            x_dec,
            mask_dec,
            self_attention_cache,
        )
        x_dec = residual + self.self_mha_dropout(self_mha_output)

        residual = x_dec
        x_dec = self.cross_mha_norm(x_dec)
        cross_mha_output, cross_attention_cache = self.cross_mha.forward_with_static_cache(
            x_dec,
            x_enc,
            x_enc,
            mask_enc,
            cross_attention_cache,
        )
        x_dec = residual + self.cross_mha_dropout(cross_mha_output)

        residual = x_dec
        x_dec = self.ffn_norm(x_dec)
        x_dec = residual + self.ffn_dropout(self.ffn(x_dec))
        return x_dec, DecoderLayerCache(
            self_attention=self_attention_cache,
            cross_attention=cross_attention_cache,
        )

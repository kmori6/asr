import torch
import torch.nn as nn

from asr.modules.transformer.cache import KVCache
from asr.modules.transformer.multi_head_attention import MultiHeadAttention


class EncoderLayer(nn.Module):
    """Pre-normalized Transformer encoder layer."""

    def __init__(
        self,
        mha: MultiHeadAttention,
        mha_norm: nn.Module,
        ffn: nn.Module,
        ffn_norm: nn.Module,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.mha = mha
        self.mha_norm = mha_norm
        self.mha_dropout = nn.Dropout(dropout_rate)
        self.ffn = ffn
        self.ffn_norm = ffn_norm
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Embedding tensor (batch_size, sequence_length, hidden_size).
            mask (torch.Tensor): Attention mask (batch_size, sequence_length, sequence_length).

        Returns:
            torch.Tensor: Output tensor with the same shape as ``x``.
        """
        residual = x
        x = self.mha_norm(x)
        x = residual + self.mha_dropout(self.mha(x, x, x, mask))

        residual = x
        x = self.ffn_norm(x)
        x = residual + self.ffn_dropout(self.ffn(x))
        return x

    def forward_with_cache(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        """Apply the layer while extending its self-attention KV cache."""
        residual = x
        x = self.mha_norm(x)
        attention, cache = self.mha.forward_with_cache(x, x, x, mask, cache)
        x = residual + self.mha_dropout(attention)

        residual = x
        x = self.ffn_norm(x)
        x = residual + self.ffn_dropout(self.ffn(x))
        return x, cache

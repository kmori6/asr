from typing import cast

import torch
import torch.nn as nn

from asr.modules.conformer.cache import ConformerBlockCache
from asr.modules.conformer.convolution import CausalConvolution, Convolution
from asr.modules.conformer.feed_forward import FeedForward
from asr.modules.conformer.multi_head_self_attention import MultiHeadSelfAttention


class ConformerBlock(nn.Module):
    """Non-streaming Conformer block.

    Proposed in A. Gulati et al., "Conformer: Convolution-augmented Transformer for Speech Recognition,"
    in Proc. Interspeech, 2020, pp. 5036-5040.

    """

    _convolution_type: type[Convolution] = Convolution

    def __init__(
        self,
        input_size: int,
        num_heads: int,
        kernel_size: int,
        dropout_rate: float,
        feed_forward_expansion_factor: int = 4,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if feed_forward_expansion_factor <= 0:
            raise ValueError("feed_forward_expansion_factor must be positive")

        hidden_size = input_size * feed_forward_expansion_factor
        self.ffn1_norm = nn.LayerNorm(input_size)
        self.ffn1 = FeedForward(input_size, hidden_size, dropout_rate, bias=bias)
        self.mha_norm = nn.LayerNorm(input_size)
        self.mha = MultiHeadSelfAttention(input_size, num_heads, dropout_rate, bias=bias)
        self.conv_norm = nn.LayerNorm(input_size)
        self.conv = self._convolution_type(input_size, kernel_size, bias=bias)
        self.ffn2_norm = nn.LayerNorm(input_size)
        self.ffn2 = FeedForward(input_size, hidden_size, dropout_rate, bias=bias)
        self.dropout = nn.Dropout(dropout_rate)
        self.final_norm = nn.LayerNorm(input_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Input tensor with shape ``(batch, num_frames, input_size)``.
            mask (torch.Tensor): Boolean attention mask with shape
                ``(batch, num_frames, num_frames)`` where ``True`` marks an
                allowed query-key pair.

        Returns:
            torch.Tensor: Output tensor with the same shape as ``x``. Invalid frames are zero.
        """
        frame_mask = mask.any(dim=-1)
        x = x + 0.5 * self.dropout(self.ffn1(self.ffn1_norm(x)))
        x = x + self.dropout(self.mha(self.mha_norm(x), mask))
        x = x + self.dropout(self.conv(self.conv_norm(x), frame_mask))
        x = x + 0.5 * self.dropout(self.ffn2(self.ffn2_norm(x)))
        x = self.final_norm(x)
        return x.masked_fill(~frame_mask[:, :, None], 0.0)


class StreamingConformerBlock(ConformerBlock):
    """Conformer block with causal convolution and cached chunk processing."""

    _convolution_type = CausalConvolution

    def forward_chunk(
        self, x: torch.Tensor, mask: torch.Tensor, cache: ConformerBlockCache | None = None
    ) -> tuple[torch.Tensor, ConformerBlockCache]:
        """

        Args:
            x (torch.Tensor): Current input chunk with shape ``(batch, chunk_size, input_size)``.
            mask (torch.Tensor): Boolean attention mask with shape
                ``(batch, chunk_size, cached_length + chunk_size)`` where ``True``
                marks an allowed query-key pair.
            cache (ConformerBlockCache | None, optional): Attention and convolution states
                from preceding chunks. ``None`` starts a new stream.

        Returns:
            tuple[torch.Tensor, ConformerBlockCache]: Current chunk outputs with shape
                ``(batch, chunk_size, input_size)`` and the updated block cache.
        """
        frame_mask = mask.any(dim=-1)
        attention_cache = None if cache is None else cache.attention
        convolution_cache = None if cache is None else cache.convolution

        x = x + 0.5 * self.dropout(self.ffn1(self.ffn1_norm(x)))
        attention_output, next_attention_cache = self.mha.forward_chunk(
            self.mha_norm(x),
            mask,
            attention_cache,
        )
        x = x + self.dropout(attention_output)
        convolution = cast(CausalConvolution, self.conv)
        convolution_output, next_convolution_cache = convolution.forward_chunk(
            self.conv_norm(x),
            frame_mask,
            convolution_cache,
        )
        x = x + self.dropout(convolution_output)
        x = x + 0.5 * self.dropout(self.ffn2(self.ffn2_norm(x)))
        x = self.final_norm(x)
        x = x.masked_fill(~frame_mask[:, :, None], 0.0)

        next_cache = ConformerBlockCache(
            attention=next_attention_cache,
            convolution=next_convolution_cache,
        )
        return x, next_cache

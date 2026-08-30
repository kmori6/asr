import math
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn

from asr.modules.transformer.cache import KVCache
from asr.modules.transformer.encoder_layer import EncoderLayer
from asr.modules.transformer.feed_forward import FeedForward
from asr.modules.transformer.multi_head_attention import MultiHeadAttention
from asr.modules.transformer.positional_encoding import PositionalEncoding


@dataclass(frozen=True)
class TransformerLMCache:
    """Self-attention caches for all Transformer LM layers."""

    layers: tuple[KVCache, ...]


class TransformerLM(nn.Module):
    """Causal Transformer language model for training and ASR decoding."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        feed_forward_size: int,
        dropout_rate: float,
        max_length: int,
        bias: bool = False,
        ignore_index: int = -100,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if max_length <= 0:
            raise ValueError("max_length must be positive")

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.position = PositionalEncoding(hidden_size, max_length)
        self.dropout = nn.Dropout(dropout_rate)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    mha=MultiHeadAttention(hidden_size, num_heads, dropout_rate, bias=bias),
                    mha_norm=nn.LayerNorm(hidden_size),
                    ffn=FeedForward(
                        hidden_size,
                        feed_forward_size,
                        dropout_rate,
                        activation=nn.GELU(),
                        bias=bias,
                    ),
                    ffn_norm=nn.LayerNorm(hidden_size),
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=label_smoothing)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """

        Args:
            input_ids (torch.Tensor): Token IDs with shape ``(batch, sequence_length)``.
            attention_mask (torch.Tensor | None, optional): Valid-token mask with the same shape as
                ``input_ids``. ``None`` treats every token as valid.
            labels (torch.Tensor | None, optional): Targets with the same shape as ``input_ids``.
                Padding must be ``-100``. ``None`` omits loss computation.

        Returns:
            dict[str, torch.Tensor]: Mapping containing ``logits`` with shape
                ``(batch, sequence_length, vocab_size)`` and an optional scalar ``loss``.
        """
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have shape (batch, sequence_length) with at least one token")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")

        valid_tokens = attention_mask.to(dtype=torch.bool)
        sequence_length = input_ids.shape[1]
        causal_mask = torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=input_ids.device).tril()
        mask = valid_tokens[:, :, None] & valid_tokens[:, None, :] & causal_mask[None, :, :]

        x = self.embedding(input_ids) * math.sqrt(self.hidden_size)
        x = self.dropout(x + self.position(x))
        for layer in self.layers:
            x = layer(x, mask)
        logits = nn.functional.linear(self.final_norm(x), self.embedding.weight)

        output = {"logits": logits}
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            output["loss"] = self.loss_fn(
                logits[:, :-1].reshape(-1, self.vocab_size),
                labels[:, 1:].reshape(-1),
            )
        return output

    @torch.inference_mode()
    def predict(
        self,
        input_ids: torch.Tensor,
        cache: TransformerLMCache | None = None,
    ) -> tuple[torch.Tensor, TransformerLMCache]:
        """Predict following-token logits and update the autoregressive KV cache.

        ``input_ids`` contains only tokens not already represented by ``cache``. The first call
        should contain the tokenizer's BOS token.
        """
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have shape (batch, sequence_length) with at least one token")
        if cache is not None and len(cache.layers) != len(self.layers):
            raise ValueError(f"cache must contain {len(self.layers)} layers, but got {len(cache.layers)}")

        cached_length = 0 if cache is None else cache.layers[0].key.shape[2]
        query_length = input_ids.shape[1]
        key_positions = torch.arange(cached_length + query_length, device=input_ids.device)
        query_positions = cached_length + torch.arange(query_length, device=input_ids.device)
        mask = key_positions[None, :] <= query_positions[:, None]
        mask = mask[None, :, :].expand(input_ids.shape[0], -1, -1)

        x = self.embedding(input_ids) * math.sqrt(self.hidden_size)
        x = self.dropout(x + self.position(x, offset=cached_length))
        layer_caches = cache.layers if cache is not None else (None,) * len(self.layers)
        next_caches = []
        for layer_module, layer_cache in zip(self.layers, layer_caches, strict=True):
            layer = cast(EncoderLayer, layer_module)
            x, next_cache = layer.forward_with_cache(x, mask, layer_cache)
            next_caches.append(next_cache)

        logits = nn.functional.linear(self.final_norm(x), self.embedding.weight)
        return logits, TransformerLMCache(layers=tuple(next_caches))

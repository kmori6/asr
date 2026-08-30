import torch
import torch.nn as nn

from asr.modules.transformer.cache import KVCache


class MultiHeadAttention(nn.Module):
    """Multi-head attention module.

    Proposed in A. Vaswani et al., "Attention is all you need," in NeurIPS, 2017, pp. 5998-6008.

    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_rate: float,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate must satisfy 0 <= dropout_rate < 1")

        self.hidden_size = hidden_size
        self.h = num_heads
        self.d_k = hidden_size // num_heads
        self.dropout_rate = dropout_rate
        self.w_q = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.w_k = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.w_v = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.w_o = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """

        Args:
            q (torch.Tensor): Query tensor (batch_size, target_sequence_length, hidden_size).
            k (torch.Tensor): Key tensor (batch_size, source_sequence_length, hidden_size).
            v (torch.Tensor): Value tensor (batch_size, source_sequence_length, hidden_size).
            mask (torch.Tensor): Mask tensor (batch_size, target_sequence_length, source_sequence_length).

        Returns:
            torch.Tensor: Output tensor (batch_size, target_sequence_length, hidden_size).
        """
        q = self._split_heads(self.w_q(q))
        k = self._split_heads(self.w_k(k))
        v = self._split_heads(self.w_v(v))
        return self._attention(q, k, v, mask)

    def forward_with_cache(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
        cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        """Compute self-attention while appending projected keys and values to a cache.

        Args:
            q (torch.Tensor): Query tensor (batch_size, target_sequence_length, hidden_size).
            k (torch.Tensor): New key tensor (batch_size, source_sequence_length, hidden_size).
            v (torch.Tensor): New value tensor (batch_size, source_sequence_length, hidden_size).
            mask (torch.Tensor): Mask tensor covering both cached and new keys
                (batch_size, target_sequence_length, cached_length + source_sequence_length).
            cache (KVCache | None): Previously projected keys and values.

        Returns:
            tuple[torch.Tensor, KVCache]: Attention output and the updated projected KV cache.
        """
        q = self._split_heads(self.w_q(q))
        k = self._split_heads(self.w_k(k))
        v = self._split_heads(self.w_v(v))
        if cache is not None:
            k = torch.cat([cache.key, k], dim=2)
            v = torch.cat([cache.value, v], dim=2)

        return self._attention(q, k, v, mask), KVCache(key=k, value=v)

    def forward_with_static_cache(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
        cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        """Compute cross-attention while reusing a cache for an unchanged source sequence.

        This is intended for decoder cross-attention, where the encoder keys and values remain
        unchanged across autoregressive decoding steps.

        Args:
            q (torch.Tensor): Query tensor (batch_size, target_sequence_length, hidden_size).
            k (torch.Tensor): Key tensor (batch_size, source_sequence_length, hidden_size).
            v (torch.Tensor): Value tensor (batch_size, source_sequence_length, hidden_size).
            mask (torch.Tensor): Mask tensor
                (batch_size, target_sequence_length, source_sequence_length).
            cache (KVCache | None): Previously projected source keys and values. When provided,
                ``k`` and ``v`` are not projected again.

        Returns:
            tuple[torch.Tensor, KVCache]: Attention output and the projected source KV cache.
        """
        q = self._split_heads(self.w_q(q))
        if cache is None:
            cached_k = self._split_heads(self.w_k(k))
            cached_v = self._split_heads(self.w_v(v))
            cache = KVCache(key=cached_k, value=cached_v)
        else:
            cached_k, cached_v = cache.key, cache.value

        return self._attention(q, cached_k, cached_v, mask), cache

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split the hidden dimension into attention heads.

        Args:
            x (torch.Tensor): Projected tensor (batch_size, sequence_length, hidden_size).

        Returns:
            torch.Tensor: Tensor (batch_size, num_heads, sequence_length, d_k).
        """
        batch_size, sequence_length = x.shape[:2]
        return x.view(batch_size, sequence_length, self.h, self.d_k).transpose(1, 2)

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Scaled dot-product attention.

        Args:
            q (torch.Tensor): Query tensor (batch_size, num_heads, target_sequence_length, d_k).
            k (torch.Tensor): Key tensor (batch_size, num_heads, source_sequence_length, d_k).
            v (torch.Tensor): Value tensor (batch_size, num_heads, source_sequence_length, d_k).
            mask (torch.Tensor): Mask tensor (batch_size, target_sequence_length, source_sequence_length).

        Returns:
            torch.Tensor: Attention output tensor (batch_size, target_sequence_length, hidden_size).
        """
        x = nn.functional.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=mask[:, None, :, :],
            dropout_p=self.dropout_rate if self.training else 0.0,
            is_causal=False,
        )
        x = x.transpose(1, 2).flatten(2, -1)
        x = self.w_o(x)
        return x

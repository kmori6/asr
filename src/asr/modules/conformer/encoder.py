from typing import cast

import torch
import torch.nn as nn

from asr.modules.conformer.attention_mask import chunk_attention_mask
from asr.modules.conformer.block import ConformerBlock
from asr.modules.conformer.cache import ConformerBlockCache, FastConformerEncoderCache
from asr.modules.conformer.subsampling import FastConformerSubsampling


class FastConformerEncoder(nn.Module):
    """FastConformer encoder trained with dynamic chunkwise attention."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_heads: int,
        kernel_size: int,
        num_blocks: int,
        dropout_rate: float,
        min_chunk_size: int,
        max_chunk_size: int,
        streaming_mask_probability: float = 0.6,
        conv_channels: int = 256,
        feed_forward_expansion_factor: int = 4,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if min_chunk_size <= 0:
            raise ValueError("min_chunk_size must be positive")
        if max_chunk_size < min_chunk_size:
            raise ValueError("max_chunk_size must be greater than or equal to min_chunk_size")
        if not 0.0 <= streaming_mask_probability <= 1.0:
            raise ValueError("streaming_mask_probability must satisfy 0 <= probability <= 1")

        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.streaming_mask_probability = streaming_mask_probability
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.subsampling = FastConformerSubsampling(input_size, hidden_size, conv_channels)
        self.dropout = nn.Dropout(dropout_rate)
        self.blocks = nn.ModuleList(
            ConformerBlock(
                input_size=hidden_size,
                num_heads=num_heads,
                kernel_size=kernel_size,
                dropout_rate=dropout_rate,
                feed_forward_expansion_factor=feed_forward_expansion_factor,
                bias=bias,
            )
            for _ in range(num_blocks)
        )

    def forward(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        chunk_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: Acoustic features with shape
                ``(batch, num_frames, input_size)``.
            feature_lengths: Valid input frame counts with shape ``(batch,)``.
            chunk_size: Chunk size measured after 8x subsampling. During
                training, omitting it applies a sampled streaming chunk size with
                ``streaming_mask_probability`` and otherwise uses full context.
                During evaluation, omitting it uses ``max_chunk_size``.

        Returns:
            Encoded features with shape
            ``(batch, ceil(num_frames / 8), hidden_size)`` and their valid
            lengths. Invalid output frames are zero.
        """
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        x, output_lengths = self.subsampling(features, feature_lengths)
        x = self.dropout(x)

        if chunk_size is None and self.training:
            # Section 2.1 uses dynamic chunk masks for 60% of batches
            # and full context for the remaining 40%: https://arxiv.org/pdf/2306.08175
            use_streaming_mask = self.streaming_mask_probability == 1.0 or (
                self.streaming_mask_probability > 0.0 and torch.rand(()).item() < self.streaming_mask_probability
            )
            if use_streaming_mask:
                chunk_size = int(torch.randint(self.min_chunk_size, self.max_chunk_size + 1, ()).item())
        elif chunk_size is None:
            chunk_size = self.max_chunk_size

        if chunk_size is None:
            chunk_size = x.shape[1]

        mask = chunk_attention_mask(output_lengths, chunk_size, max_length=x.shape[1])
        for block in self.blocks:
            x = block(x, mask)

        return x, output_lengths

    def forward_chunk(
        self,
        features: torch.Tensor,
        cache: FastConformerEncoderCache | None = None,
        chunk_size: int | None = None,
        is_final: bool = False,
    ) -> tuple[torch.Tensor, FastConformerEncoderCache]:
        """
        Args:
            features: Next unpadded feature chunk for one utterance with shape
                ``(1, num_frames, input_size)``. Every frame is treated as valid.
            cache: Encoder states returned by the preceding call, or ``None``
                for the first chunk.
            chunk_size: Attention chunk size after 8x subsampling. ``None`` uses
                ``max_chunk_size`` and the value must remain fixed per utterance.
            is_final: If ``True``, also encode a final incomplete attention chunk.

        Returns:
            Newly available encoder frames and the updated streaming cache.
            A non-final call may return zero frames while an attention chunk is
            being accumulated.
        """
        if features.ndim != 3 or features.shape[0] != 1:
            raise ValueError(
                f"streaming features must have shape (1, num_frames, input_size), but got {tuple(features.shape)}"
            )
        if features.shape[-1] != self.input_size:
            raise ValueError(f"expected input_size {self.input_size}, but got {features.shape[-1]}")
        if chunk_size is None:
            chunk_size = self.max_chunk_size
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        block_caches: tuple[ConformerBlockCache, ...]
        if cache is None:
            subsampling_cache = None
            pending = None
            block_caches = ()
        else:
            self._validate_streaming_cache(cache, features, chunk_size)
            subsampling_cache = cache.subsampling
            pending = cache.pending
            block_caches = cache.blocks

        x, next_subsampling_cache = self.subsampling.forward_chunk(features, subsampling_cache)
        x = self.dropout(x)
        if pending is None:
            pending = x.new_empty((1, 0, self.hidden_size))
        elif pending.dtype != x.dtype:
            raise ValueError("new and pending encoder frames must have the same dtype")
        pending = torch.cat((pending, x), dim=1)

        num_ready_frames = pending.shape[1] if is_final else pending.shape[1] // chunk_size * chunk_size
        ready = pending[:, :num_ready_frames]
        next_pending = pending[:, num_ready_frames:]
        encoded_chunks: list[torch.Tensor] = []
        next_block_caches = block_caches

        for current_chunk in ready.split(chunk_size, dim=1):
            if current_chunk.shape[1] == 0:
                continue
            current_block_caches = next_block_caches
            updated_block_caches: list[ConformerBlockCache] = []
            for index, module in enumerate(self.blocks):
                block = cast(ConformerBlock, module)
                block_cache = None if not current_block_caches else current_block_caches[index]
                cached_length = 0 if block_cache is None else block_cache.attention.key.shape[2]
                mask = torch.ones(
                    (1, current_chunk.shape[1], cached_length + current_chunk.shape[1]),
                    dtype=torch.bool,
                    device=current_chunk.device,
                )
                current_chunk, next_block_cache = block.forward_chunk(current_chunk, mask, block_cache)
                updated_block_caches.append(next_block_cache)
            next_block_caches = tuple(updated_block_caches)
            encoded_chunks.append(current_chunk)

        if encoded_chunks:
            outputs = torch.cat(encoded_chunks, dim=1)
        else:
            outputs = pending.new_empty((1, 0, self.hidden_size))

        next_cache = FastConformerEncoderCache(
            subsampling=next_subsampling_cache,
            pending=next_pending,
            blocks=next_block_caches,
            chunk_size=chunk_size,
        )
        return outputs, next_cache

    def _validate_streaming_cache(
        self,
        cache: FastConformerEncoderCache,
        features: torch.Tensor,
        chunk_size: int,
    ) -> None:
        if cache.chunk_size != chunk_size:
            raise ValueError("chunk_size must remain fixed while decoding one utterance")
        if cache.pending.ndim != 3 or cache.pending.shape[0] != 1 or cache.pending.shape[2] != self.hidden_size:
            raise ValueError("pending encoder frames must have shape (1, pending_length, hidden_size)")
        if cache.pending.shape[1] >= chunk_size:
            raise ValueError("pending encoder frame count must be smaller than chunk_size")
        if cache.pending.device != features.device:
            raise ValueError("streaming features and pending encoder frames must be on the same device")
        if cache.blocks and len(cache.blocks) != len(self.blocks):
            raise ValueError(f"encoder cache must contain {len(self.blocks)} block states")

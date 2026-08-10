import torch
import torch.nn as nn

from asr.modules.conformer.attention_mask import chunk_attention_mask
from asr.modules.conformer.block import ConformerBlock
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

import torch


def chunk_attention_mask(
    lengths: torch.Tensor,
    chunk_size: int,
    max_length: int | None = None,
) -> torch.Tensor:
    """Create a chunkwise self-attention mask.

    Proposed in X. Chen et al., "Developing realtime streaming transformer transducer for speech recognition
    on large-scale dataset," in ICASSP, 2021, pp. 5904-5908.

    Each query can attend to every valid key in preceding chunks and its
    complete current chunk. This permits up to ``chunk_size - 1`` frames of
    lookahead while preventing attention to future chunks.

    Args:
        lengths: Valid sequence lengths with shape ``(batch,)``.
        chunk_size: Number of frames in each attention chunk.
        max_length: Padded sequence length. Defaults to the largest value in
            ``lengths``.

    Returns:
        Boolean tensor with shape ``(batch, max_length, max_length)`` where
        ``True`` marks an allowed query-key pair. Padded queries and keys are
        entirely masked out.
    """
    if lengths.ndim != 1:
        raise ValueError(f"lengths must have shape (batch,), but got {tuple(lengths.shape)}")
    if lengths.numel() == 0:
        raise ValueError("lengths must contain at least one sequence")
    if lengths.dtype == torch.bool or torch.is_floating_point(lengths):
        raise ValueError("lengths must have an integer dtype")
    if torch.any(lengths < 0):
        raise ValueError("lengths must be non-negative")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if max_length is None:
        max_length = int(lengths.max().item())
    if max_length < 0:
        raise ValueError("max_length must be non-negative")
    if torch.any(lengths > max_length):
        raise ValueError("max_length must be greater than or equal to every sequence length")

    positions = torch.arange(max_length, device=lengths.device)
    valid_frames = positions[None, :] < lengths[:, None]
    chunk_ends = (positions // chunk_size + 1) * chunk_size
    within_visible_chunks = positions[None, :] < chunk_ends[:, None]

    return valid_frames[:, :, None] & valid_frames[:, None, :] & within_visible_chunks[None, :, :]

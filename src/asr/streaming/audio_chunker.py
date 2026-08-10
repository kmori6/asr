from dataclasses import dataclass
from typing import Iterator

import torch


@dataclass(frozen=True, slots=True)
class AudioChunk:
    """One non-overlapping waveform chunk and its final-chunk marker."""

    waveform: torch.Tensor
    is_final: bool


class AudioChunker:
    """Split a mono waveform into non-overlapping streaming chunks."""

    def __init__(self, chunk_duration_ms: int, sample_rate: int) -> None:
        if chunk_duration_ms <= 0:
            raise ValueError("chunk_duration_ms must be positive")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

        self.chunk_duration_ms = chunk_duration_ms
        self.sample_rate = sample_rate
        self.chunk_size = round(chunk_duration_ms * sample_rate / 1_000)
        if self.chunk_size == 0:
            raise ValueError("chunk_duration_ms is too short to contain one sample")

    def stream(self, waveform: torch.Tensor) -> Iterator[AudioChunk]:
        """
        Args:
            waveform: Non-empty mono waveform with shape ``(num_samples,)``.

        Yields:
            Consecutive views of at most ``chunk_size`` samples. Only the last
            chunk has ``is_final=True``; chunks are not padded or overlapped.
        """
        if waveform.ndim != 1:
            raise ValueError(f"waveform must have shape (num_samples,), but got {tuple(waveform.shape)}")
        if not waveform.is_floating_point():
            raise TypeError("waveform must be a floating-point tensor")
        if waveform.numel() == 0:
            raise ValueError("waveform must contain at least one sample")

        for start in range(0, waveform.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, waveform.shape[0])
            yield AudioChunk(waveform=waveform[start:end], is_final=end == waveform.shape[0])

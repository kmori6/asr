import torch

from asr.decoding import CTCBeamSearch, CTCBeamSearchResult, RNNTBeamSearch, RNNTBeamSearchResult
from asr.models import (
    StreamingFastConformerCTC,
    StreamingFastConformerCTCCache,
    StreamingFastConformerRNNT,
    StreamingFastConformerRNNTCache,
)
from asr.streaming.audio_chunker import AudioChunker


class StreamingRNNTRecognizer:
    """Run cache-aware FastConformer RNN-T recognition on waveform chunks."""

    def __init__(
        self,
        model: StreamingFastConformerRNNT,
        searcher: RNNTBeamSearch,
        chunk_size: int,
        language_model_weight: float = 0.0,
        amp_dtype: torch.dtype | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if language_model_weight < 0.0:
            raise ValueError("language_model_weight must be non-negative")
        if searcher.prediction_network is not model.prediction_network:
            raise ValueError("searcher and model must share the same prediction network")
        if searcher.joint_network is not model.joint_network:
            raise ValueError("searcher and model must share the same joint network")

        self.model = model
        self.searcher = searcher
        self.chunk_size = chunk_size
        self.language_model_weight = language_model_weight
        self.amp_dtype = amp_dtype

    @torch.inference_mode()
    def recognize(self, waveform: torch.Tensor, chunker: AudioChunker) -> RNNTBeamSearchResult:
        """
        Args:
            waveform: Complete mono waveform with shape ``(num_samples,)``.
                Chunks are revealed sequentially by ``chunker``.
            chunker: Non-overlapping waveform chunk iterator.

        Returns:
            Best hypothesis after the final chunk.
        """
        if self.model.training:
            raise RuntimeError("streaming recognition requires model.eval()")

        device = next(self.model.parameters()).device
        self.searcher.reset()
        cache: StreamingFastConformerRNNTCache | None = None
        result: RNNTBeamSearchResult | None = None

        for chunk in chunker.stream(waveform):
            chunk_waveform = chunk.waveform.to(device=device).unsqueeze(0)
            with torch.autocast(
                device_type=device.type,
                dtype=self.amp_dtype,
                enabled=device.type == "cuda" and self.amp_dtype is not None,
            ):
                encoder_outputs, cache = self.model.encode_chunk(
                    chunk_waveform,
                    cache=cache,
                    chunk_size=self.chunk_size,
                    is_final=chunk.is_final,
                )
                result = self.searcher.search(
                    encoder_outputs,
                    language_model_weight=self.language_model_weight,
                )

        if result is None:
            raise RuntimeError("audio chunker did not yield any chunks")
        return result


class StreamingCTCRecognizer:
    """Run cache-aware FastConformer CTC recognition on waveform chunks."""

    def __init__(
        self,
        model: StreamingFastConformerCTC,
        searcher: CTCBeamSearch,
        chunk_size: int,
        language_model_weight: float = 0.0,
        amp_dtype: torch.dtype | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if language_model_weight < 0.0:
            raise ValueError("language_model_weight must be non-negative")
        if searcher.blank_token_id != model.blank_token_id:
            raise ValueError("searcher and model must use the same blank token ID")

        self.model = model
        self.searcher = searcher
        self.chunk_size = chunk_size
        self.language_model_weight = language_model_weight
        self.amp_dtype = amp_dtype

    @torch.inference_mode()
    def recognize(self, waveform: torch.Tensor, chunker: AudioChunker) -> CTCBeamSearchResult:
        """Decode one waveform while retaining encoder, CTC, and LM states."""
        if self.model.training:
            raise RuntimeError("streaming recognition requires model.eval()")

        device = next(self.model.parameters()).device
        self.searcher.reset()
        cache: StreamingFastConformerCTCCache | None = None
        result: CTCBeamSearchResult | None = None

        for chunk in chunker.stream(waveform):
            chunk_waveform = chunk.waveform.to(device=device).unsqueeze(0)
            with torch.autocast(
                device_type=device.type,
                dtype=self.amp_dtype,
                enabled=device.type == "cuda" and self.amp_dtype is not None,
            ):
                encoder_outputs, cache = self.model.encode_chunk(
                    chunk_waveform,
                    cache=cache,
                    chunk_size=self.chunk_size,
                    is_final=chunk.is_final,
                )
                logits = self.model.logits(encoder_outputs)[0]
                result = self.searcher.search_chunk(
                    logits,
                    language_model_weight=self.language_model_weight,
                )

        if result is None:
            raise RuntimeError("audio chunker did not yield any chunks")
        return result

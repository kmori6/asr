import torch

from asr.decoding import RNNTBeamSearch, RNNTBeamSearchResult
from asr.models import FastConformerRNNT, FastConformerRNNTCache
from asr.streaming.audio_chunker import AudioChunker


class StreamingRecognizer:
    """Run cache-aware FastConformer RNN-T recognition on waveform chunks."""

    def __init__(
        self,
        model: FastConformerRNNT,
        searcher: RNNTBeamSearch,
        chunk_size: int,
        amp_dtype: torch.dtype | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if searcher.prediction_network is not model.prediction_network:
            raise ValueError("searcher and model must share the same prediction network")
        if searcher.joint_network is not model.joint_network:
            raise ValueError("searcher and model must share the same joint network")

        self.model = model
        self.searcher = searcher
        self.chunk_size = chunk_size
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
        cache: FastConformerRNNTCache | None = None
        result: RNNTBeamSearchResult | None = None

        for chunk in chunker.stream(waveform):
            chunk_waveform = chunk.waveform.to(device=device).unsqueeze(0)
            chunk_length = torch.tensor([chunk_waveform.shape[1]], dtype=torch.long, device=device)
            with torch.autocast(
                device_type=device.type,
                dtype=self.amp_dtype,
                enabled=device.type == "cuda" and self.amp_dtype is not None,
            ):
                encoder_outputs, cache = self.model.encode_chunk(
                    chunk_waveform,
                    chunk_length,
                    cache=cache,
                    chunk_size=self.chunk_size,
                    is_final=chunk.is_final,
                )
                result = self.searcher.search(encoder_outputs)

        if result is None:
            raise RuntimeError("audio chunker did not yield any chunks")
        return result

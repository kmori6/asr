import torch

from asr.streaming import AudioChunker


def test_audio_chunker_yields_non_overlapping_chunks() -> None:
    waveform = torch.arange(10, dtype=torch.float32)
    chunker = AudioChunker(chunk_duration_ms=4, sample_rate=1_000)

    chunks = list(chunker.stream(waveform))

    assert [chunk.waveform.shape[0] for chunk in chunks] == [4, 4, 2]
    assert [chunk.is_final for chunk in chunks] == [False, False, True]
    torch.testing.assert_close(torch.cat([chunk.waveform for chunk in chunks]), waveform)

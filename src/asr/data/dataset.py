import json
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SpeechTextSample:
    """One speech-recognition sample.

    Attributes:
        utterance_id: Unique utterance identifier from the manifest.
        waveform: Mono waveform with shape ``(num_samples,)`` and dtype ``torch.float32``.
        sample_rate: Waveform sample rate in Hz.
        text: Normalized reference transcript.
    """

    utterance_id: str
    waveform: torch.Tensor
    sample_rate: int
    text: str


@dataclass(frozen=True)
class _ManifestRecord:
    utterance_id: str
    audio_path: Path
    num_samples: int
    sample_rate: int
    text: str


class SpeechTextDataset(Dataset[SpeechTextSample]):
    """Load waveforms and transcripts from a JSON manifest.

    Relative audio paths are resolved from the directory containing the manifest.
    Audio is required to be mono and to match the sample rate recorded in the manifest.
    """

    def __init__(self, manifest_path: str | Path, sample_rate: int = 16_000) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

        self.manifest_path = Path(manifest_path).expanduser().resolve()
        with self.manifest_path.open(encoding="utf-8") as manifest_file:
            raw_records: object = json.load(manifest_file)
        if not isinstance(raw_records, list):
            raise ValueError(f"Manifest must contain a JSON list: {self.manifest_path}")

        self.sample_rate = sample_rate
        self._records = [self._parse_record(record, index) for index, record in enumerate(raw_records)]

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> SpeechTextSample:
        record = self._records[index]
        waveform, sample_rate = sf.read(record.audio_path, dtype="float32", always_2d=True)

        if waveform.shape[1] != 1:
            raise ValueError(f"Expected mono audio, but got {waveform.shape[1]} channels: {record.audio_path}")
        if sample_rate != record.sample_rate:
            raise ValueError(
                f"Expected a {record.sample_rate} Hz sample rate, but got {sample_rate}: {record.audio_path}"
            )
        if waveform.shape[0] != record.num_samples:
            raise ValueError(f"Expected {record.num_samples} samples, but got {waveform.shape[0]}: {record.audio_path}")

        return SpeechTextSample(
            utterance_id=record.utterance_id,
            waveform=torch.from_numpy(waveform[:, 0]),
            sample_rate=sample_rate,
            text=record.text,
        )

    def _parse_record(self, value: object, index: int) -> _ManifestRecord:
        if not isinstance(value, dict):
            raise ValueError(f"Manifest record {index} must be a JSON object")

        utterance_id = value.get("id")
        audio_path = value.get("audio_path")
        num_samples = value.get("num_samples")
        sample_rate = value.get("sample_rate")
        text = value.get("text")

        if not isinstance(utterance_id, str) or not utterance_id:
            raise ValueError(f"Manifest record {index} has an invalid id")
        if not isinstance(audio_path, str) or not audio_path:
            raise ValueError(f"Manifest record {index} has an invalid audio_path")
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or num_samples <= 0:
            raise ValueError(f"Manifest record {index} has an invalid num_samples")
        if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
            raise ValueError(f"Manifest record {index} has an invalid sample_rate")
        if sample_rate != self.sample_rate:
            raise ValueError(f"Manifest record {index} has sample rate {sample_rate}, expected {self.sample_rate}")
        if not isinstance(text, str):
            raise ValueError(f"Manifest record {index} has invalid text")

        resolved_audio_path = Path(audio_path).expanduser()
        if not resolved_audio_path.is_absolute():
            resolved_audio_path = self.manifest_path.parent / resolved_audio_path

        return _ManifestRecord(
            utterance_id=utterance_id,
            audio_path=resolved_audio_path,
            num_samples=num_samples,
            sample_rate=sample_rate,
            text=text,
        )

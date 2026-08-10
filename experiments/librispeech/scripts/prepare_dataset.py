import argparse
import json
import os
import unicodedata
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

SAMPLE_RATE = 16_000
SPLIT_TO_SUBSETS = {
    "train": ("train-clean-100", "train-clean-360", "train-other-500"),
    "valid": ("dev-clean", "dev-other"),
    "test-clean": ("test-clean",),
    "test-other": ("test-other",),
}


def normalize_transcript(text: str) -> str:
    """Normalize a LibriSpeech transcript for tokenizer training and WER evaluation."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


def read_subset(
    subset_dir: Path,
    out_dir: Path,
    min_duration: float | None,
    max_duration: float | None,
) -> list[dict[str, str | int]]:
    """Read metadata for one extracted LibriSpeech subset."""
    if not subset_dir.is_dir():
        raise FileNotFoundError(f"LibriSpeech subset not found: {subset_dir}")

    records: list[dict[str, str | int]] = []
    transcript_paths = sorted(subset_dir.rglob("*.trans.txt"))
    if not transcript_paths:
        raise FileNotFoundError(f"No transcript files found under: {subset_dir}")

    for transcript_path in tqdm(transcript_paths, desc=subset_dir.name, leave=False):
        with transcript_path.open(encoding="utf-8") as transcript_file:
            for line in transcript_file:
                utterance_id, transcript = line.rstrip("\n").split(maxsplit=1)
                audio_path = transcript_path.parent / f"{utterance_id}.flac"
                if not audio_path.is_file():
                    raise FileNotFoundError(f"Audio file not found: {audio_path}")

                audio_info = sf.info(audio_path)
                if audio_info.channels != 1:
                    raise ValueError(f"Expected mono audio, but got {audio_info.channels} channels: {audio_path}")
                if audio_info.samplerate != SAMPLE_RATE:
                    raise ValueError(
                        f"Expected a {SAMPLE_RATE} Hz sample rate, but got {audio_info.samplerate}: {audio_path}"
                    )

                duration = audio_info.frames / SAMPLE_RATE
                if min_duration is not None and duration < min_duration:
                    continue
                if max_duration is not None and duration > max_duration:
                    continue

                relative_audio_path = Path(os.path.relpath(audio_path.resolve(), out_dir.resolve())).as_posix()
                records.append(
                    {
                        "id": utterance_id,
                        "audio_path": relative_audio_path,
                        "num_samples": audio_info.frames,
                        "sample_rate": audio_info.samplerate,
                        "text": normalize_transcript(transcript),
                    }
                )

    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--min_duration", type=float, default=1.0)
    parser.add_argument("--max_duration", type=float, default=20.0)
    args = parser.parse_args()
    if args.min_duration < 0:
        parser.error("--min_duration must be non-negative")
    if args.max_duration <= args.min_duration:
        parser.error("--max_duration must be greater than --min_duration")

    librispeech_dir = args.data_dir.expanduser().resolve() / "LibriSpeech"
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_text: list[str] = []
    for split, subsets in SPLIT_TO_SUBSETS.items():
        records: list[dict[str, str | int]] = []
        filter_by_duration = split in {"train", "valid"}
        for subset in subsets:
            records.extend(
                read_subset(
                    subset_dir=librispeech_dir / subset,
                    out_dir=out_dir,
                    min_duration=args.min_duration if filter_by_duration else None,
                    max_duration=args.max_duration if filter_by_duration else None,
                )
            )

        with (out_dir / f"{split}.json").open("w", encoding="utf-8") as output_file:
            json.dump(records, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")
        if split == "train":
            train_text.extend(f"{record['text']}\n" for record in records)

    with (out_dir / "train.txt").open("w", encoding="utf-8") as text_file:
        text_file.writelines(train_text)


if __name__ == "__main__":
    main()

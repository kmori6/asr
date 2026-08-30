import argparse
import gzip
import json
import random
import unicodedata
from pathlib import Path

from tqdm import tqdm


def normalize_text(text: str) -> str:
    """Apply the same text normalization used by the shared tokenizer."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--valid_ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not 0.0 < args.valid_ratio < 1.0:
        parser.error("--valid_ratio must satisfy 0 < valid_ratio < 1")

    input_path = args.input_path.expanduser().resolve()
    if not input_path.is_file():
        parser.error(f"--input_path does not exist: {input_path}")

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    random_generator = random.Random(args.seed)

    split_files = {
        "train": (out_dir / "train.json").open("w", encoding="utf-8"),
        "valid": (out_dir / "valid.json").open("w", encoding="utf-8"),
    }
    counts = {"train": 0, "valid": 0}

    try:
        with gzip.open(input_path, mode="rt", encoding="utf-8") as input_file:
            for line in tqdm(input_file, desc="Preparing LM data"):
                text = normalize_text(line)
                if not text:
                    continue

                split = "valid" if random_generator.random() < args.valid_ratio else "train"
                output_file = split_files[split]
                output_file.write(f"{json.dumps({'text': text}, ensure_ascii=False)}\n")
                counts[split] += 1
    finally:
        for output_file in split_files.values():
            output_file.close()

    if counts["train"] == 0 or counts["valid"] == 0:
        raise ValueError("Both train and valid splits must contain at least one sample")

    print(f"Wrote {counts['train']} training and {counts['valid']} validation samples to {out_dir}")


if __name__ == "__main__":
    main()

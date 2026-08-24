import json
import time
from logging import getLogger
from pathlib import Path
from typing import cast

import hydra
import torch
from hubert_ctc_factory import load_hubert_ctc, recognize_hubert_ctc
from omegaconf import DictConfig

from asr.data import load_audio
from asr.decoding import CTCBeamSearch

logger = getLogger(__name__)
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]


def resolve_experiment_path(path: str) -> Path:
    """Resolve a config path relative to the LibriSpeech experiment directory."""
    resolved_path = Path(path).expanduser()
    return resolved_path if resolved_path.is_absolute() else EXPERIMENT_DIR / resolved_path


@hydra.main(version_base=None, config_path="../config", config_name="hubert_ctc")
def main(config: DictConfig) -> None:
    if config.infer.input_path is None:
        raise ValueError("set infer.input_path to an audio file")

    input_path = resolve_experiment_path(str(config.infer.input_path))
    model_path = resolve_experiment_path(str(config.infer.model_path))
    out_path = resolve_experiment_path(str(config.infer.out_path))
    if not input_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {input_path}")
    if not model_path.is_dir():
        raise FileNotFoundError(f"HuBERT model directory not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    amp_dtype = torch.bfloat16 if device.type != "cuda" or torch.cuda.is_bf16_supported() else torch.float16

    sample_rate = int(config.dataset.sample_rate)
    model, processor, tokenizer, blank_token_id = load_hubert_ctc(model_path, sample_rate, device)
    searcher = CTCBeamSearch(
        beam_width=int(config.infer.beam_size),
        blank_token_id=blank_token_id,
    )
    waveform = load_audio(input_path, sample_rate=sample_rate)

    start_time = time.perf_counter()
    result = recognize_hubert_ctc(
        waveform=waveform,
        sample_rate=sample_rate,
        model=model,
        processor=processor,
        searcher=searcher,
        device=device,
        amp_dtype=amp_dtype,
    )
    inference_seconds = time.perf_counter() - start_time

    transcript = cast(str, tokenizer.decode(result.token_ids, skip_special_tokens=True)).strip()
    audio_duration_seconds = waveform.shape[0] / sample_rate
    metadata = {
        "input_path": str(input_path.resolve()),
        "model_path": str(model_path.resolve()),
        "transcript": transcript,
        "token_ids": result.token_ids,
        "score": result.score,
        "sample_rate": sample_rate,
        "num_samples": waveform.shape[0],
        "audio_duration_seconds": audio_duration_seconds,
        "inference_seconds": inference_seconds,
        "real_time_factor": inference_seconds / audio_duration_seconds,
        "device": str(device),
        "beam_size": int(config.infer.beam_size),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as output_file:
        json.dump(metadata, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")

    print(transcript)
    logger.info("Recognized %s -> %s", input_path, out_path)


if __name__ == "__main__":
    main()

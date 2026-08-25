import json
import time
from logging import getLogger
from pathlib import Path
from typing import cast

import hydra
import torch
from omegaconf import DictConfig
from whisper_factory import is_whisper_short_form, load_whisper, recognize_whisper

from asr.data import load_audio

logger = getLogger(__name__)
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]


def resolve_experiment_path(path: str) -> Path:
    """Resolve a config path relative to the LibriSpeech experiment directory."""
    resolved_path = Path(path).expanduser()
    return resolved_path if resolved_path.is_absolute() else EXPERIMENT_DIR / resolved_path


@hydra.main(version_base=None, config_path="../config", config_name="whisper")
def main(config: DictConfig) -> None:
    if config.infer.input_path is None:
        raise ValueError("set infer.input_path to an audio file")

    input_path = resolve_experiment_path(str(config.infer.input_path))
    model_path = resolve_experiment_path(str(config.infer.model_path))
    out_path = resolve_experiment_path(str(config.infer.out_path))
    if not input_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {input_path}")
    if not model_path.is_dir():
        raise FileNotFoundError(f"Whisper model directory not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    amp_dtype = torch.bfloat16 if device.type != "cuda" or torch.cuda.is_bf16_supported() else torch.float16

    sample_rate = int(config.dataset.sample_rate)
    language = str(config.model.language)
    task = str(config.model.task)
    model, processor, tokenizer = load_whisper(model_path, sample_rate, language, task, device)
    waveform = load_audio(input_path, sample_rate=sample_rate)
    audio_duration_seconds = waveform.shape[0] / sample_rate
    max_audio_seconds = processor.feature_extractor.n_samples / sample_rate
    metadata: dict[str, object] = {
        "input_path": str(input_path.resolve()),
        "model_path": str(model_path.resolve()),
        "sample_rate": sample_rate,
        "num_samples": waveform.shape[0],
        "audio_duration_seconds": audio_duration_seconds,
        "device": str(device),
        "language": language,
        "task": task,
        "num_beams": int(config.infer.num_beams),
        "max_new_tokens": int(config.infer.max_new_tokens),
        "max_audio_seconds": max_audio_seconds,
    }

    if not is_whisper_short_form(waveform, processor):
        metadata.update(
            {
                "status": "skipped",
                "reason": f"audio exceeds the {max_audio_seconds:g}-second Whisper short-form limit",
            }
        )
        logger.warning(
            "Skipping %s because its duration %.3f seconds exceeds %.0f seconds",
            input_path,
            audio_duration_seconds,
            max_audio_seconds,
        )
    else:
        start_time = time.perf_counter()
        token_ids = recognize_whisper(
            waveform=waveform,
            sample_rate=sample_rate,
            model=model,
            processor=processor,
            language=language,
            task=task,
            num_beams=int(config.infer.num_beams),
            max_new_tokens=int(config.infer.max_new_tokens),
            device=device,
            amp_dtype=amp_dtype,
        )
        inference_seconds = time.perf_counter() - start_time
        transcript = cast(
            str,
            tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ),
        ).strip()
        metadata.update(
            {
                "status": "recognized",
                "transcript": transcript,
                "token_ids": token_ids,
                "inference_seconds": inference_seconds,
                "real_time_factor": inference_seconds / audio_duration_seconds,
            }
        )
        print(transcript)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as output_file:
        json.dump(metadata, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")

    logger.info("Wrote inference result for %s -> %s", input_path, out_path)


if __name__ == "__main__":
    main()

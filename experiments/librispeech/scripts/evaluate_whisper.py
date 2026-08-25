import json
import time
from logging import getLogger
from pathlib import Path
from typing import cast

import hydra
import torch
from omegaconf import DictConfig
from torchaudio.functional import edit_distance
from tqdm import tqdm
from transformers import WhisperTokenizer
from whisper_factory import (
    is_whisper_short_form,
    load_whisper,
    recognize_whisper,
    restore_english_spelling_normalizer,
)

from asr.data import SpeechTextDataset

logger = getLogger(__name__)
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]


def resolve_experiment_path(path: str) -> Path:
    """Resolve a config path relative to the LibriSpeech experiment directory."""
    resolved_path = Path(path).expanduser()
    return resolved_path if resolved_path.is_absolute() else EXPERIMENT_DIR / resolved_path


def word_error_rate(hypotheses: list[str], references: list[str]) -> tuple[float, int]:
    """Compute corpus WER as total word edits divided by reference words."""
    num_reference_words = sum(len(reference.split()) for reference in references)
    if num_reference_words == 0:
        raise ValueError("WER requires at least one reference word")
    num_errors = sum(
        edit_distance(reference.split(), hypothesis.split())
        for hypothesis, reference in zip(hypotheses, references, strict=True)
    )
    return num_errors / num_reference_words, num_reference_words


@hydra.main(version_base=None, config_path="../config", config_name="whisper")
def main(config: DictConfig) -> None:
    sample_rate = int(config.dataset.sample_rate)
    data_dir = resolve_experiment_path(str(config.dataset.data_dir))
    test_path = data_dir / str(config.evaluate.test_file)
    model_path = resolve_experiment_path(str(config.evaluate.model_path))
    out_dir = resolve_experiment_path(str(config.evaluate.out_dir))
    if not test_path.is_file():
        raise FileNotFoundError(f"Evaluation manifest not found: {test_path}")
    if not model_path.is_dir():
        raise FileNotFoundError(f"Whisper model directory not found: {model_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = SpeechTextDataset(test_path, sample_rate=sample_rate)
    if len(dataset) == 0:
        raise ValueError("The evaluation dataset must not be empty.")
    max_samples = config.evaluate.max_samples
    num_samples = len(dataset) if max_samples is None else min(len(dataset), int(max_samples))
    if num_samples <= 0:
        raise ValueError("evaluate.max_samples must be positive or null")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    amp_dtype = torch.bfloat16 if device.type != "cuda" or torch.cuda.is_bf16_supported() else torch.float16
    language = str(config.model.language)
    task = str(config.model.task)
    model, processor, tokenizer = load_whisper(model_path, sample_rate, language, task, device)
    if tokenizer.english_spelling_normalizer is None:
        fallback_tokenizer = WhisperTokenizer.from_pretrained(
            str(config.model.pretrained_model_name_or_path),
            language=language,
            task=task,
        )
        restore_english_spelling_normalizer(tokenizer, fallback_tokenizer)

    hypotheses: list[str] = []
    references: list[str] = []
    total_audio_seconds = 0.0
    num_skipped_utterances = 0
    max_audio_seconds = processor.feature_extractor.n_samples / sample_rate
    start_time = time.perf_counter()
    with (
        (out_dir / "ref.txt").open("w", encoding="utf-8") as reference_file,
        (out_dir / "hyp.txt").open("w", encoding="utf-8") as hypothesis_file,
        (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as predictions_file,
    ):
        for index in tqdm(range(num_samples), desc="Evaluating", dynamic_ncols=True):
            sample = dataset[index]
            audio_duration_seconds = sample.waveform.shape[0] / sample_rate
            if not is_whisper_short_form(sample.waveform, processor):
                num_skipped_utterances += 1
                json.dump(
                    {
                        "id": sample.utterance_id,
                        "reference": sample.text,
                        "status": "skipped",
                        "reason": f"audio exceeds the {max_audio_seconds:g}-second Whisper short-form limit",
                        "audio_duration_seconds": audio_duration_seconds,
                    },
                    predictions_file,
                    ensure_ascii=False,
                )
                predictions_file.write("\n")
                logger.warning(
                    "Skipping %s because its duration %.3f seconds exceeds %.0f seconds",
                    sample.utterance_id,
                    audio_duration_seconds,
                    max_audio_seconds,
                )
                continue

            token_ids = recognize_whisper(
                waveform=sample.waveform,
                sample_rate=sample_rate,
                model=model,
                processor=processor,
                language=language,
                task=task,
                num_beams=int(config.evaluate.num_beams),
                max_new_tokens=int(config.evaluate.max_new_tokens),
                device=device,
                amp_dtype=amp_dtype,
            )
            raw_hypothesis = cast(
                str,
                tokenizer.decode(
                    token_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ),
            ).strip()
            reference = tokenizer.normalize(sample.text).strip()
            hypothesis = tokenizer.normalize(raw_hypothesis).strip()
            references.append(reference)
            hypotheses.append(hypothesis)
            total_audio_seconds += audio_duration_seconds
            reference_file.write(f"{reference}\n")
            hypothesis_file.write(f"{hypothesis}\n")
            json.dump(
                {
                    "id": sample.utterance_id,
                    "status": "evaluated",
                    "reference": sample.text,
                    "hypothesis": raw_hypothesis,
                    "normalized_reference": reference,
                    "normalized_hypothesis": hypothesis,
                    "token_ids": token_ids,
                },
                predictions_file,
                ensure_ascii=False,
            )
            predictions_file.write("\n")

    evaluation_seconds = time.perf_counter() - start_time
    if not references:
        raise ValueError(f"No evaluation utterances fit within the {max_audio_seconds:g}-second short-form limit")
    wer, num_reference_words = word_error_rate(hypotheses, references)
    num_evaluated_utterances = len(references)
    metrics = {
        "wer": wer,
        "num_utterances": num_evaluated_utterances,
        "num_requested_utterances": num_samples,
        "num_skipped_utterances": num_skipped_utterances,
        "max_audio_seconds": max_audio_seconds,
        "num_reference_words": num_reference_words,
        "audio_duration_seconds": total_audio_seconds,
        "evaluation_seconds": evaluation_seconds,
        "real_time_factor": evaluation_seconds / total_audio_seconds,
        "device": str(device),
        "language": language,
        "task": task,
        "num_beams": int(config.evaluate.num_beams),
        "max_new_tokens": int(config.evaluate.max_new_tokens),
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, ensure_ascii=False, indent=2)
        metrics_file.write("\n")
    logger.info(
        "WER: %.4f (%d evaluated, %d skipped utterances)",
        wer,
        num_evaluated_utterances,
        num_skipped_utterances,
    )


if __name__ == "__main__":
    main()

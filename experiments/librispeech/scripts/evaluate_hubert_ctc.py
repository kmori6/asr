import json
import time
from logging import getLogger
from pathlib import Path
from typing import cast

import hydra
import torch
from hubert_ctc_factory import load_hubert_ctc, recognize_hubert_ctc
from omegaconf import DictConfig
from torchaudio.functional import edit_distance
from tqdm import tqdm

from asr.data import SpeechTextDataset
from asr.decoding import CTCBeamSearch

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


@hydra.main(version_base=None, config_path="../config", config_name="hubert_ctc")
def main(config: DictConfig) -> None:
    sample_rate = int(config.dataset.sample_rate)
    data_dir = resolve_experiment_path(str(config.dataset.data_dir))
    test_path = data_dir / str(config.evaluate.test_file)
    model_path = resolve_experiment_path(str(config.evaluate.model_path))
    out_dir = resolve_experiment_path(str(config.evaluate.out_dir))
    if not test_path.is_file():
        raise FileNotFoundError(f"Evaluation manifest not found: {test_path}")
    if not model_path.is_dir():
        raise FileNotFoundError(f"HuBERT model directory not found: {model_path}")
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

    model, processor, tokenizer, blank_token_id = load_hubert_ctc(model_path, sample_rate, device)
    searcher = CTCBeamSearch(
        beam_width=int(config.evaluate.beam_size),
        blank_token_id=blank_token_id,
    )

    hypotheses: list[str] = []
    references: list[str] = []
    total_audio_seconds = 0.0
    start_time = time.perf_counter()
    with (
        (out_dir / "ref.txt").open("w", encoding="utf-8") as reference_file,
        (out_dir / "hyp.txt").open("w", encoding="utf-8") as hypothesis_file,
        (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as predictions_file,
    ):
        for index in tqdm(range(num_samples), desc="Evaluating", dynamic_ncols=True):
            sample = dataset[index]
            result = recognize_hubert_ctc(
                waveform=sample.waveform,
                sample_rate=sample_rate,
                model=model,
                processor=processor,
                searcher=searcher,
                device=device,
                amp_dtype=amp_dtype,
            )
            hypothesis = cast(str, tokenizer.decode(result.token_ids, skip_special_tokens=True)).strip()
            references.append(sample.text)
            hypotheses.append(hypothesis)
            total_audio_seconds += sample.waveform.shape[0] / sample_rate
            reference_file.write(f"{sample.text}\n")
            hypothesis_file.write(f"{hypothesis}\n")
            json.dump(
                {
                    "id": sample.utterance_id,
                    "reference": sample.text,
                    "hypothesis": hypothesis,
                    "token_ids": result.token_ids,
                    "score": result.score,
                },
                predictions_file,
                ensure_ascii=False,
            )
            predictions_file.write("\n")

    evaluation_seconds = time.perf_counter() - start_time
    wer, num_reference_words = word_error_rate(hypotheses, references)
    metrics = {
        "wer": wer,
        "num_utterances": num_samples,
        "num_reference_words": num_reference_words,
        "audio_duration_seconds": total_audio_seconds,
        "evaluation_seconds": evaluation_seconds,
        "real_time_factor": evaluation_seconds / total_audio_seconds,
        "device": str(device),
        "beam_size": int(config.evaluate.beam_size),
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, ensure_ascii=False, indent=2)
        metrics_file.write("\n")
    logger.info("WER: %.4f (%d utterances)", wer, num_samples)


if __name__ == "__main__":
    main()

import json
from logging import getLogger
from pathlib import Path
from typing import cast

import hydra
import torch
from fast_conformer_rnnt_factory import (
    build_fast_conformer_rnnt,
    load_model_weights,
    load_transformer_lm,
    validate_tokenizer,
)
from omegaconf import DictConfig
from torchaudio.functional import edit_distance
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from asr.data import SpeechTextDataset
from asr.decoding import RNNTBeamSearch
from asr.models import FastConformerRNNT

logger = getLogger(__name__)
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]


def resolve_experiment_path(path: str) -> Path:
    """Resolve a config path relative to the LibriSpeech experiment directory."""
    resolved_path = Path(path).expanduser()
    return resolved_path if resolved_path.is_absolute() else EXPERIMENT_DIR / resolved_path


@torch.inference_mode()
def recognize(
    waveform: torch.Tensor,
    model: FastConformerRNNT,
    searcher: RNNTBeamSearch,
    device: torch.device,
    amp_dtype: torch.dtype,
    language_model_weight: float,
) -> list[int]:
    """Encode one complete waveform and return its best token sequence."""
    searcher.reset()
    waveform = waveform.to(device).unsqueeze(0)
    waveform_length = torch.tensor([waveform.shape[1]], dtype=torch.long, device=device)
    with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
        encoder_outputs, encoder_lengths = model.encode(waveform, waveform_length)
        num_encoder_frames = int(encoder_lengths[0].item())
        return searcher.search(
            encoder_outputs[:, :num_encoder_frames],
            language_model_weight=language_model_weight,
        ).token_ids


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


@hydra.main(version_base=None, config_path="../config", config_name="fast_conformer_rnnt")
def main(config: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    amp_dtype = torch.bfloat16 if device.type != "cuda" or torch.cuda.is_bf16_supported() else torch.float16

    data_dir = resolve_experiment_path(config.dataset.data_dir)
    test_path = data_dir / config.evaluate.test_file
    tokenizer_dir = resolve_experiment_path(config.tokenizer.tokenizer_dir)
    model_path = resolve_experiment_path(config.evaluate.model_path)
    out_dir = resolve_experiment_path(config.evaluate.out_dir)
    for required_path in (test_path, tokenizer_dir, model_path):
        if not required_path.exists():
            raise FileNotFoundError(f"Required evaluation input not found: {required_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = SpeechTextDataset(test_path, sample_rate=config.frontend.sample_rate)
    if len(dataset) == 0:
        raise ValueError("The evaluation dataset must not be empty.")
    num_samples = (
        len(dataset) if config.evaluate.max_samples is None else min(len(dataset), config.evaluate.max_samples)
    )
    if num_samples <= 0:
        raise ValueError("evaluate.max_samples must be positive or null")

    tokenizer = cast(
        PreTrainedTokenizerFast,
        PreTrainedTokenizerFast.from_pretrained(tokenizer_dir),
    )
    blank_token_id, bos_token_id, eos_token_id = validate_tokenizer(tokenizer, int(config.model.vocab_size))
    model = build_fast_conformer_rnnt(config, blank_token_id).to(device)
    load_model_weights(model, model_path, device)
    model.eval()
    language_model_weight = float(config.evaluate.language_model_weight)
    if language_model_weight < 0.0:
        raise ValueError("evaluate.language_model_weight must be non-negative")
    language_model_path = resolve_experiment_path(str(config.language_model.model_path))
    language_model = (
        load_transformer_lm(language_model_path, tokenizer, device) if language_model_weight > 0.0 else None
    )
    searcher = RNNTBeamSearch(
        prediction_network=model.prediction_network,
        joint_network=model.joint_network,
        beam_width=int(config.evaluate.beam_size),
        blank_token_id=blank_token_id,
        language_model=language_model,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
    )

    hypotheses: list[str] = []
    references: list[str] = []
    with (
        (out_dir / "ref.txt").open("w", encoding="utf-8") as reference_file,
        (out_dir / "hyp.txt").open("w", encoding="utf-8") as hypothesis_file,
        (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as predictions_file,
    ):
        for index in tqdm(range(num_samples), desc="Evaluating", dynamic_ncols=True):
            sample = dataset[index]
            token_ids = recognize(
                sample.waveform,
                model,
                searcher,
                device,
                amp_dtype,
                language_model_weight,
            )
            hypothesis = cast(str, tokenizer.decode(token_ids, skip_special_tokens=True)).strip()
            references.append(sample.text)
            hypotheses.append(hypothesis)
            reference_file.write(f"{sample.text}\n")
            hypothesis_file.write(f"{hypothesis}\n")
            json.dump(
                {"id": sample.utterance_id, "reference": sample.text, "hypothesis": hypothesis},
                predictions_file,
                ensure_ascii=False,
            )
            predictions_file.write("\n")

    wer, num_reference_words = word_error_rate(hypotheses, references)
    metrics = {
        "wer": wer,
        "num_utterances": num_samples,
        "num_reference_words": num_reference_words,
        "streaming": False,
        "beam_size": int(config.evaluate.beam_size),
        "language_model_weight": language_model_weight,
        "language_model_path": str(language_model_path) if language_model is not None else None,
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, ensure_ascii=False, indent=2)
        metrics_file.write("\n")
    logger.info("WER: %.4f (%d utterances)", wer, num_samples)


if __name__ == "__main__":
    main()

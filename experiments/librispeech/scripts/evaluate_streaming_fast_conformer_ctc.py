import json
from logging import getLogger
from typing import cast

import hydra
import torch
from omegaconf import DictConfig
from torchaudio.functional import edit_distance
from tqdm import tqdm
from train_streaming_fast_conformer_ctc import (
    build_streaming_fast_conformer_ctc,
    load_model_weights,
    load_transformer_lm,
    resolve_experiment_path,
    validate_tokenizer,
)
from transformers import PreTrainedTokenizerFast

from asr.data import SpeechTextDataset
from asr.decoding import CTCBeamSearch
from asr.streaming import AudioChunker, StreamingCTCRecognizer

logger = getLogger(__name__)


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


@hydra.main(version_base=None, config_path="../config", config_name="streaming_fast_conformer_ctc")
def main(config: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    amp_dtype = torch.bfloat16 if device.type != "cuda" or torch.cuda.is_bf16_supported() else torch.float16

    data_dir = resolve_experiment_path(str(config.dataset.data_dir))
    test_path = data_dir / str(config.evaluate.test_file)
    tokenizer_dir = resolve_experiment_path(str(config.tokenizer.tokenizer_dir))
    model_path = resolve_experiment_path(str(config.evaluate.model_path))
    out_dir = resolve_experiment_path(str(config.evaluate.out_dir))
    for required_path in (test_path, tokenizer_dir, model_path):
        if not required_path.exists():
            raise FileNotFoundError(f"Required evaluation input not found: {required_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = SpeechTextDataset(test_path, sample_rate=int(config.frontend.sample_rate))
    if len(dataset) == 0:
        raise ValueError("The evaluation dataset must not be empty.")
    max_samples = config.evaluate.max_samples
    num_samples = len(dataset) if max_samples is None else min(len(dataset), int(max_samples))
    if num_samples <= 0:
        raise ValueError("evaluate.max_samples must be positive or null")

    tokenizer = cast(PreTrainedTokenizerFast, PreTrainedTokenizerFast.from_pretrained(tokenizer_dir))
    blank_token_id, bos_token_id, eos_token_id = validate_tokenizer(tokenizer, int(config.model.vocab_size))
    model = build_streaming_fast_conformer_ctc(config, blank_token_id).to(device)
    load_model_weights(model, model_path, device)
    model.eval()

    language_model_weight = float(config.evaluate.language_model_weight)
    if language_model_weight < 0.0:
        raise ValueError("evaluate.language_model_weight must be non-negative")
    language_model_path = resolve_experiment_path(str(config.language_model.model_path))
    language_model = (
        load_transformer_lm(language_model_path, tokenizer, device) if language_model_weight > 0.0 else None
    )
    searcher = CTCBeamSearch(
        beam_width=int(config.evaluate.beam_size),
        blank_token_id=blank_token_id,
        language_model=language_model,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
    )
    recognizer = StreamingCTCRecognizer(
        model=model,
        searcher=searcher,
        chunk_size=int(config.evaluate.chunk_size),
        language_model_weight=language_model_weight,
        amp_dtype=amp_dtype,
    )
    audio_chunker = AudioChunker(
        chunk_duration_ms=int(config.evaluate.audio_chunk_duration_ms),
        sample_rate=int(config.frontend.sample_rate),
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
            result = recognizer.recognize(sample.waveform, audio_chunker)
            hypothesis = cast(str, tokenizer.decode(result.token_ids, skip_special_tokens=True)).strip()
            hypotheses.append(hypothesis)
            references.append(sample.text)
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

    wer, num_reference_words = word_error_rate(hypotheses, references)
    metrics = {
        "wer": wer,
        "num_utterances": num_samples,
        "num_reference_words": num_reference_words,
        "streaming": True,
        "beam_size": int(config.evaluate.beam_size),
        "encoder_chunk_size": int(config.evaluate.chunk_size),
        "audio_chunk_duration_ms": int(config.evaluate.audio_chunk_duration_ms),
        "language_model_weight": language_model_weight,
        "language_model_path": str(language_model_path) if language_model is not None else None,
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, ensure_ascii=False, indent=2)
        metrics_file.write("\n")
    logger.info("WER: %.4f (%d utterances)", wer, num_samples)


if __name__ == "__main__":
    main()

import json
import re
import time
from logging import getLogger
from pathlib import Path
from typing import cast

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torchaudio.functional import edit_distance
from tqdm import tqdm
from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor

from asr.data import SpeechTextDataset, WavlmQwen3Collator
from asr.models import WavLMQwen3

logger = getLogger(__name__)
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
WORD_PATTERN = re.compile(r"[A-Z0-9]+(?:'[A-Z0-9]+)*")


def resolve_experiment_path(path: str) -> Path:
    """Resolve a config path relative to the LibriSpeech experiment directory."""
    resolved_path = Path(path).expanduser()
    return resolved_path if resolved_path.is_absolute() else EXPERIMENT_DIR / resolved_path


def select_inference_dtype(value: str, device: torch.device) -> torch.dtype:
    """Return the model and autocast dtype for an inference precision policy."""
    if value == "auto":
        if device.type != "cuda":
            return torch.float32
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if value == "fp32":
        return torch.float32
    if value == "bf16":
        if device.type != "cuda" or not torch.cuda.is_bf16_supported():
            raise ValueError("BF16 inference requires a CUDA device with BF16 support.")
        return torch.bfloat16
    if value == "fp16":
        if device.type != "cuda":
            raise ValueError("FP16 inference requires a CUDA device.")
        return torch.float16
    raise ValueError("mixed_precision must be one of: auto, bf16, fp16, fp32")


def normalize_librispeech_text(text: str) -> str:
    """Normalize text to uppercase LibriSpeech words without punctuation."""
    return " ".join(WORD_PATTERN.findall(text.upper()))


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


@hydra.main(version_base=None, config_path="../config", config_name="wavlm_qwen3")
def main(config: DictConfig) -> None:
    sample_rate = int(config.dataset.sample_rate)
    data_dir = resolve_experiment_path(str(config.dataset.data_dir))
    test_path = data_dir / str(config.evaluate.test_file)
    model_path = resolve_experiment_path(str(config.evaluate.model_path))
    out_dir = resolve_experiment_path(str(config.evaluate.out_dir))
    if not test_path.is_file():
        raise FileNotFoundError(f"Evaluation manifest not found: {test_path}")
    if not model_path.is_dir():
        raise FileNotFoundError(f"WavLM-Qwen3 model directory not found: {model_path}")
    checkpoint_config_path = model_path / "resolved_config.yaml"
    state_path = model_path / "pytorch_model.bin"
    if not checkpoint_config_path.is_file() or not state_path.is_file():
        raise FileNotFoundError(f"Incomplete WavLM-Qwen3 checkpoint directory: {model_path}")
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
    model_dtype = select_inference_dtype(str(config.evaluate.mixed_precision), device)
    checkpoint_config = OmegaConf.load(checkpoint_config_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path / "tokenizer")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path / "feature_extractor")
    collator = WavlmQwen3Collator(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        sample_rate=sample_rate,
        language=str(checkpoint_config.model.language),
        max_text_length=int(checkpoint_config.model.max_text_length),
    )
    model = WavLMQwen3.from_pretrained(
        speech_encoder_name_or_path=str(checkpoint_config.model.speech_encoder_name_or_path),
        language_model_name_or_path=str(checkpoint_config.model.language_model_name_or_path),
        audio_downsample_factor=int(checkpoint_config.model.audio_downsample_factor),
        dtype=model_dtype,
    )
    llm_adaptation = str(checkpoint_config.model.llm.adaptation)
    if llm_adaptation == "lora":
        model.add_language_model_lora(
            rank=int(checkpoint_config.model.llm.lora.rank),
            alpha=int(checkpoint_config.model.llm.lora.alpha),
            dropout=float(checkpoint_config.model.llm.lora.dropout),
            target_modules=[str(name) for name in checkpoint_config.model.llm.lora.target_modules],
        )
    elif llm_adaptation != "frozen":
        raise ValueError("Saved model.llm.adaptation must be one of: lora, frozen")
    state_dict = cast(
        dict[str, torch.Tensor],
        torch.load(state_path, map_location="cpu", weights_only=True, mmap=True),
    )
    model.load_state_dict(state_dict)
    model.to(device).eval()

    generation_input_ids = collator.create_generation_input_ids().to(device)
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
            waveform = sample.waveform.to(device=device).unsqueeze(0)
            waveform_lengths = torch.tensor([sample.waveform.shape[0]], dtype=torch.long, device=device)
            with torch.autocast(
                device_type=device.type,
                dtype=model_dtype,
                enabled=device.type == "cuda" and model_dtype != torch.float32,
            ):
                generated_ids = model.generate(
                    waveforms=waveform,
                    waveform_lengths=waveform_lengths,
                    input_ids=generation_input_ids,
                    max_new_tokens=int(config.evaluate.max_new_tokens),
                    num_beams=int(config.evaluate.num_beams),
                )[0]
            raw_response, raw_hypothesis = collator.decode_response(generated_ids.cpu())
            reference = normalize_librispeech_text(sample.text)
            hypothesis = normalize_librispeech_text(raw_hypothesis)
            references.append(reference)
            hypotheses.append(hypothesis)
            total_audio_seconds += sample.waveform.shape[0] / sample_rate
            reference_file.write(f"{reference}\n")
            hypothesis_file.write(f"{hypothesis}\n")
            json.dump(
                {
                    "id": sample.utterance_id,
                    "reference": sample.text,
                    "raw_response": raw_response,
                    "hypothesis": raw_hypothesis,
                    "normalized_reference": reference,
                    "normalized_hypothesis": hypothesis,
                    "response_prefix_matched": raw_response.startswith(collator.response_prefix),
                    "token_ids": generated_ids.tolist(),
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
        "num_beams": int(config.evaluate.num_beams),
        "max_new_tokens": int(config.evaluate.max_new_tokens),
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, ensure_ascii=False, indent=2)
        metrics_file.write("\n")
    logger.info("WER: %.4f (%d utterances)", wer, num_samples)


if __name__ == "__main__":
    main()

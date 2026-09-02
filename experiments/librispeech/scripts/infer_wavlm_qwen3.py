import json
import time
from logging import getLogger
from pathlib import Path
from typing import cast

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor

from asr.data import WavlmQwen3Collator, load_audio
from asr.models import WavLMQwen3
from asr.streaming import AudioChunker

logger = getLogger(__name__)
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]


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


def common_prefix_length(left: torch.Tensor, right: torch.Tensor) -> int:
    """Return the number of identical leading token IDs in two one-dimensional sequences."""
    common_length = min(left.shape[0], right.shape[0])
    if common_length == 0:
        return 0
    differences = left[:common_length].ne(right[:common_length]).nonzero()
    return common_length if differences.numel() == 0 else int(differences[0].item())


@hydra.main(version_base=None, config_path="../config", config_name="wavlm_qwen3")
def main(config: DictConfig) -> None:
    if config.infer.input_path is None:
        raise ValueError("set infer.input_path to an audio file")

    input_path = resolve_experiment_path(str(config.infer.input_path))
    model_path = resolve_experiment_path(str(config.infer.model_path))
    out_path = resolve_experiment_path(str(config.infer.out_path))
    if not input_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {input_path}")
    if not model_path.is_dir():
        raise FileNotFoundError(f"WavLM-Qwen3 model directory not found: {model_path}")
    checkpoint_config_path = model_path / "resolved_config.yaml"
    state_path = model_path / "pytorch_model.bin"
    if not checkpoint_config_path.is_file() or not state_path.is_file():
        raise FileNotFoundError(f"Incomplete WavLM-Qwen3 checkpoint directory: {model_path}")

    mode = str(config.infer.mode)
    if mode not in {"offline", "streaming"}:
        raise ValueError("infer.mode must be one of: offline, streaming")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    model_dtype = select_inference_dtype(str(config.infer.mixed_precision), device)

    checkpoint_config = OmegaConf.load(checkpoint_config_path)
    sample_rate = int(checkpoint_config.dataset.sample_rate)
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

    waveform = load_audio(input_path, sample_rate=sample_rate)
    generation_input_ids = collator.create_generation_input_ids().to(device)
    chunker = AudioChunker(int(config.infer.chunk_duration_ms), sample_rate)
    chunks = [waveform] if mode == "offline" else [chunk.waveform for chunk in chunker.stream(waveform)]
    accumulated_waveform = torch.empty(0, dtype=waveform.dtype)
    previous_token_ids = torch.empty(0, dtype=torch.long)
    updates: list[dict[str, object]] = []
    total_start_time = time.perf_counter()
    for index, chunk in enumerate(chunks):
        accumulated_waveform = torch.cat((accumulated_waveform, chunk))
        device_waveform = accumulated_waveform.to(device=device).unsqueeze(0)
        waveform_lengths = torch.tensor([accumulated_waveform.shape[0]], dtype=torch.long, device=device)
        update_start_time = time.perf_counter()
        with torch.autocast(
            device_type=device.type,
            dtype=model_dtype,
            enabled=device.type == "cuda" and model_dtype != torch.float32,
        ):
            generated_ids = model.generate(
                waveforms=device_waveform,
                waveform_lengths=waveform_lengths,
                input_ids=generation_input_ids,
                max_new_tokens=int(config.infer.max_new_tokens),
                num_beams=int(config.infer.num_beams),
            )[0].cpu()
        update_seconds = time.perf_counter() - update_start_time
        raw_response, transcript = collator.decode_response(generated_ids)
        is_final = index == len(chunks) - 1
        stable_token_count = (
            generated_ids.shape[0] if is_final else common_prefix_length(previous_token_ids, generated_ids)
        )
        _, stable_transcript = collator.decode_response(generated_ids[:stable_token_count])
        updates.append(
            {
                "audio_duration_seconds": accumulated_waveform.shape[0] / sample_rate,
                "is_final": is_final,
                "raw_response": raw_response,
                "transcript": transcript,
                "stable_transcript": stable_transcript,
                "stable_token_count": stable_token_count,
                "token_ids": generated_ids.tolist(),
                "inference_seconds": update_seconds,
            }
        )
        previous_token_ids = generated_ids
        if mode == "streaming":
            print(transcript)

    total_inference_seconds = time.perf_counter() - total_start_time
    final_update = updates[-1]
    if mode == "offline":
        print(final_update["transcript"])
    metadata = {
        "input_path": str(input_path.resolve()),
        "model_path": str(model_path.resolve()),
        "mode": mode,
        "sample_rate": sample_rate,
        "num_samples": waveform.shape[0],
        "audio_duration_seconds": waveform.shape[0] / sample_rate,
        "device": str(device),
        "num_beams": int(config.infer.num_beams),
        "max_new_tokens": int(config.infer.max_new_tokens),
        "chunk_duration_ms": int(config.infer.chunk_duration_ms) if mode == "streaming" else None,
        "transcript": final_update["transcript"],
        "token_ids": final_update["token_ids"],
        "total_inference_seconds": total_inference_seconds,
        "real_time_factor": total_inference_seconds / (waveform.shape[0] / sample_rate),
        "updates": updates,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as output_file:
        json.dump(metadata, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    logger.info("Wrote %s inference result for %s -> %s", mode, input_path, out_path)


if __name__ == "__main__":
    main()

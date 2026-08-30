import json
import time
from logging import getLogger
from pathlib import Path
from typing import cast

import hydra
import torch
from fast_conformer_rnnt_factory import build_fast_conformer_rnnt, load_model_weights, validate_tokenizer
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerFast

from asr.data import load_audio
from asr.decoding import RNNTBeamSearch, RNNTBeamSearchResult
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
) -> RNNTBeamSearchResult:
    """Encode one complete waveform and return its best RNN-T hypothesis."""
    searcher.reset()
    batched_waveform = waveform.to(device).unsqueeze(0)
    waveform_length = torch.tensor([batched_waveform.shape[1]], dtype=torch.long, device=device)
    with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
        encoder_outputs, encoder_lengths = model.encode(batched_waveform, waveform_length)
        num_encoder_frames = int(encoder_lengths[0].item())
        return searcher.search(encoder_outputs[:, :num_encoder_frames])


@hydra.main(version_base=None, config_path="../config", config_name="fast_conformer_rnnt")
def main(config: DictConfig) -> None:
    if config.infer.input_path is None:
        raise ValueError("set infer.input_path to an audio file")

    input_path = resolve_experiment_path(str(config.infer.input_path))
    model_path = resolve_experiment_path(str(config.infer.model_path))
    tokenizer_dir = resolve_experiment_path(str(config.tokenizer.tokenizer_dir))
    out_path = resolve_experiment_path(str(config.infer.out_path))
    for required_path in (input_path, model_path, tokenizer_dir):
        if not required_path.exists():
            raise FileNotFoundError(f"Required inference input not found: {required_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    amp_dtype = torch.bfloat16 if device.type != "cuda" or torch.cuda.is_bf16_supported() else torch.float16

    tokenizer = cast(
        PreTrainedTokenizerFast,
        PreTrainedTokenizerFast.from_pretrained(tokenizer_dir),
    )
    blank_token_id = validate_tokenizer(tokenizer, int(config.model.vocab_size))
    model = build_fast_conformer_rnnt(config, blank_token_id).to(device)
    load_model_weights(model, model_path, device)
    model.eval()
    searcher = RNNTBeamSearch(
        prediction_network=model.prediction_network,
        joint_network=model.joint_network,
        beam_width=int(config.infer.beam_size),
        blank_token_id=blank_token_id,
    )
    sample_rate = int(config.frontend.sample_rate)
    waveform = load_audio(input_path, sample_rate=sample_rate)

    start_time = time.perf_counter()
    result = recognize(waveform, model, searcher, device, amp_dtype)
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
        "streaming": False,
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

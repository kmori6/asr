from pathlib import Path
from typing import cast

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer


def configure_whisper_generation(
    model: WhisperForConditionalGeneration,
    language: str,
    task: str,
) -> None:
    """Configure current Whisper language/task fields and clear their legacy equivalent."""
    model.config.forced_decoder_ids = None
    # These Whisper-specific fields are loaded dynamically onto GenerationConfig,
    # whose base-class type annotations do not declare them.
    setattr(model.generation_config, "forced_decoder_ids", None)
    setattr(model.generation_config, "language", language)
    setattr(model.generation_config, "task", task)
    setattr(model.generation_config, "is_multilingual", True)


def restore_english_spelling_normalizer(
    tokenizer: WhisperTokenizer,
    fallback_tokenizer: WhisperTokenizer,
) -> None:
    """Restore Whisper's English spelling map when it is absent from a saved processor."""
    if tokenizer.english_spelling_normalizer is not None:
        return
    if fallback_tokenizer.english_spelling_normalizer is None:
        raise ValueError("The fallback Whisper tokenizer does not contain an English spelling normalizer.")
    tokenizer.english_spelling_normalizer = fallback_tokenizer.english_spelling_normalizer


def is_whisper_short_form(waveform: torch.Tensor, processor: WhisperProcessor) -> bool:
    """Return whether a waveform fits in one Whisper 30-second feature window."""
    return waveform.shape[0] <= processor.feature_extractor.n_samples


def load_whisper(
    model_path: str | Path,
    sample_rate: int,
    language: str,
    task: str,
    device: torch.device,
) -> tuple[WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer]:
    """Load and validate a fine-tuned Whisper model and processor."""
    processor = WhisperProcessor.from_pretrained(model_path, language=language, task=task)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    tokenizer = cast(WhisperTokenizer, processor.tokenizer)

    if processor.feature_extractor.sampling_rate != sample_rate:
        raise ValueError(
            f"Whisper processor requires {processor.feature_extractor.sampling_rate} Hz audio, "
            f"but dataset.sample_rate is {sample_rate}."
        )
    decoder_start_token_id = model.config.decoder_start_token_id
    if decoder_start_token_id is None:
        raise ValueError("Whisper model must define decoder_start_token_id.")
    if not tokenizer.prefix_tokens or tokenizer.prefix_tokens[0] != decoder_start_token_id:
        raise ValueError("Whisper tokenizer prefix must begin with the model decoder_start_token_id.")

    configure_whisper_generation(model, language, task)
    model.config.use_cache = True
    torch.nn.Module.to(model, device)
    model.eval()
    return model, processor, tokenizer


@torch.inference_mode()
def recognize_whisper(
    waveform: torch.Tensor,
    sample_rate: int,
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    language: str,
    task: str,
    num_beams: int,
    max_new_tokens: int,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> list[int]:
    """Generate one short-form Whisper transcription from a mono waveform."""
    if waveform.ndim != 1 or waveform.numel() == 0:
        raise ValueError("waveform must have shape (num_samples,) with at least one sample")
    if not torch.isfinite(waveform).all():
        raise ValueError("waveform must contain only finite values")
    if not is_whisper_short_form(waveform, processor):
        max_duration_seconds = processor.feature_extractor.n_samples / sample_rate
        raise ValueError(f"Whisper inference audio must not exceed {max_duration_seconds:g} seconds")
    if num_beams < 1:
        raise ValueError("num_beams must be positive")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")

    tokenizer = cast(WhisperTokenizer, processor.tokenizer)
    max_length = len(tokenizer.prefix_tokens) + max_new_tokens
    if max_length > model.config.max_target_positions:
        raise ValueError(
            f"Whisper prefix and generated tokens must not exceed {model.config.max_target_positions} tokens, "
            f"but got {max_length}."
        )

    encoded = processor.feature_extractor(
        waveform.detach().cpu().numpy(),
        sampling_rate=sample_rate,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_features = cast(torch.Tensor, encoded["input_features"]).to(device)
    attention_mask = cast(torch.Tensor, encoded["attention_mask"]).to(device)
    with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
        generated_token_ids = cast(
            torch.Tensor,
            model.generate(
                input_features=input_features,
                attention_mask=attention_mask,
                language=language,
                task=task,
                is_multilingual=True,
                return_timestamps=False,
                num_beams=num_beams,
                max_length=max_length,
            ),
        )
    return cast(list[int], generated_token_ids[0].tolist())

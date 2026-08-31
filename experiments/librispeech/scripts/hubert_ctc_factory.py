from pathlib import Path
from typing import cast

import torch
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file
from transformers import (
    HubertForCTC,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    Wav2Vec2Processor,
)

from asr.decoding import CTCBeamSearch, CTCBeamSearchResult
from asr.models import TransformerLM


def configure_tokenizer(
    tokenizer: PreTrainedTokenizerFast,
    blank_token: str,
    expected_vocab_size: int,
) -> int:
    """Validate the experiment tokenizer and register its CTC blank as padding."""
    if len(tokenizer) != expected_vocab_size:
        raise ValueError(f"Tokenizer vocabulary size must be {expected_vocab_size}, but got {len(tokenizer)}.")
    vocabulary = tokenizer.get_vocab()
    if blank_token not in vocabulary:
        raise ValueError(f"Tokenizer must define the CTC blank token {blank_token}.")
    if tokenizer.unk_token_id is None:
        raise ValueError("Tokenizer must define unk_token.")

    tokenizer.pad_token = blank_token
    blank_token_id = vocabulary[blank_token]
    if tokenizer.pad_token_id != blank_token_id:
        raise ValueError("The tokenizer pad token must use the CTC blank token ID.")
    if blank_token_id == tokenizer.unk_token_id:
        raise ValueError("CTC blank and unknown tokens must have different IDs.")
    return blank_token_id


def load_hubert_ctc(
    model_path: str | Path,
    sample_rate: int,
    device: torch.device,
) -> tuple[HubertForCTC, Wav2Vec2Processor, PreTrainedTokenizerBase, int]:
    """Load and validate a fine-tuned HuBERT CTC model and its processor."""
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model = HubertForCTC.from_pretrained(model_path)
    tokenizer = cast(PreTrainedTokenizerBase, processor.tokenizer)

    if processor.feature_extractor.sampling_rate != sample_rate:
        raise ValueError(
            f"HuBERT processor requires {processor.feature_extractor.sampling_rate} Hz audio, "
            f"but dataset.sample_rate is {sample_rate}."
        )
    blank_token_id = tokenizer.pad_token_id
    if blank_token_id is None:
        raise ValueError("The saved HuBERT tokenizer must define its CTC blank as pad_token.")
    if model.config.pad_token_id != blank_token_id:
        raise ValueError(
            "The HuBERT model and tokenizer disagree about the CTC blank token ID: "
            f"{model.config.pad_token_id} != {blank_token_id}."
        )
    if model.config.vocab_size != len(tokenizer):
        raise ValueError(
            f"The HuBERT model and tokenizer vocabulary sizes differ: {model.config.vocab_size} != {len(tokenizer)}."
        )
    torch.nn.Module.to(model, device)
    model.eval()
    return model, processor, tokenizer, blank_token_id


def load_transformer_lm(
    model_dir: Path,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
) -> TransformerLM:
    """Load the Transformer LM and verify its tokenizer matches the CTC tokenizer."""
    weights_path = model_dir / "model.safetensors"
    config_path = model_dir / "config.yaml"
    for required_path in (weights_path, config_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"Required language-model file not found: {required_path}")

    language_model_tokenizer = cast(
        PreTrainedTokenizerFast,
        PreTrainedTokenizerFast.from_pretrained(model_dir),
    )
    if language_model_tokenizer.get_vocab() != tokenizer.get_vocab():
        raise ValueError("Language-model and CTC tokenizers must have the same vocabulary and token IDs")
    language_model_special_ids = (
        language_model_tokenizer.pad_token_id,
        language_model_tokenizer.bos_token_id,
        language_model_tokenizer.eos_token_id,
    )
    ctc_special_ids = (tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id)
    if language_model_special_ids != ctc_special_ids:
        raise ValueError("Language-model and CTC tokenizers must use the same PAD, BOS, and EOS token IDs")

    saved_config = cast(DictConfig, OmegaConf.load(config_path))
    model_config = saved_config.model
    if int(model_config.vocab_size) != len(tokenizer):
        raise ValueError("Language-model vocabulary size must match the CTC tokenizer")
    language_model = TransformerLM(
        vocab_size=int(model_config.vocab_size),
        hidden_size=int(model_config.hidden_size),
        num_heads=int(model_config.num_heads),
        num_layers=int(model_config.num_layers),
        feed_forward_size=int(model_config.feed_forward_size),
        dropout_rate=float(model_config.dropout_rate),
        max_length=int(model_config.max_length),
        bias=bool(model_config.bias),
    )
    language_model.load_state_dict(load_file(weights_path))
    language_model.to(device)
    language_model.eval()
    return language_model


@torch.inference_mode()
def recognize_hubert_ctc(
    waveform: torch.Tensor,
    sample_rate: int,
    model: HubertForCTC,
    processor: Wav2Vec2Processor,
    searcher: CTCBeamSearch,
    device: torch.device,
    amp_dtype: torch.dtype,
    language_model_weight: float = 0.0,
) -> CTCBeamSearchResult:
    """Compute HuBERT frame logits for one waveform and run CTC beam search."""
    if waveform.ndim != 1 or waveform.numel() == 0:
        raise ValueError("waveform must have shape (num_samples,) with at least one sample")
    if not torch.isfinite(waveform).all():
        raise ValueError("waveform must contain only finite values")

    encoded = processor(
        waveform.detach().cpu().numpy(),
        sampling_rate=sample_rate,
        return_tensors="pt",
    )
    model_inputs = {name: cast(torch.Tensor, value).to(device) for name, value in encoded.items()}
    with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
        logits = model(**model_inputs).logits[0]
    return searcher.search(logits, language_model_weight=language_model_weight)

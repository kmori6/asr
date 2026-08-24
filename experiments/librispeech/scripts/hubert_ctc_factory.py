from pathlib import Path
from typing import cast

import torch
from transformers import (
    HubertForCTC,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    Wav2Vec2Processor,
)

from asr.decoding import CTCBeamSearch, CTCBeamSearchResult


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


@torch.inference_mode()
def recognize_hubert_ctc(
    waveform: torch.Tensor,
    sample_rate: int,
    model: HubertForCTC,
    processor: Wav2Vec2Processor,
    searcher: CTCBeamSearch,
    device: torch.device,
    amp_dtype: torch.dtype,
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
    return searcher.search(logits)

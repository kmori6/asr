from pathlib import Path
from typing import cast

import torch
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file
from transformers import PreTrainedTokenizerFast

from asr.models import FastConformerTransformer, TransformerLM
from asr.modules.conformer import FastConformer
from asr.modules.frontend import LogMelSpectrogram, SpecAugment

BLANK_TOKEN = "[BLANK]"


def validate_tokenizer(
    tokenizer: PreTrainedTokenizerFast,
    expected_vocab_size: int,
) -> tuple[int, int, int]:
    """Validate and return padding, BOS, and EOS token IDs."""
    if len(tokenizer) != expected_vocab_size:
        raise ValueError(f"Tokenizer vocabulary size must be {expected_vocab_size}, but got {len(tokenizer)}.")
    if BLANK_TOKEN not in tokenizer.get_vocab():
        raise ValueError(f"Tokenizer must define {BLANK_TOKEN} for target padding.")
    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None or tokenizer.unk_token_id is None:
        raise ValueError("Tokenizer must define BOS, EOS, and UNK tokens.")

    tokenizer.pad_token = BLANK_TOKEN
    pad_token_id = cast(int, tokenizer.pad_token_id)
    bos_token_id = cast(int, tokenizer.bos_token_id)
    eos_token_id = cast(int, tokenizer.eos_token_id)
    if len({pad_token_id, bos_token_id, eos_token_id, cast(int, tokenizer.unk_token_id)}) != 4:
        raise ValueError("Padding, BOS, EOS, and UNK token IDs must be distinct.")
    return pad_token_id, bos_token_id, eos_token_id


def build_fast_conformer_transformer(config: DictConfig, blank_token_id: int) -> FastConformerTransformer:
    """Build the LibriSpeech FastConformer-Transformer model."""
    frontend = LogMelSpectrogram(
        sample_rate=config.frontend.sample_rate,
        n_fft=config.frontend.n_fft,
        hop_length=config.frontend.hop_length,
        n_mels=config.frontend.n_mels,
        f_min=config.frontend.f_min,
        f_max=config.frontend.f_max,
    )
    spec_augment = SpecAugment(
        num_frequency_masks=config.spec_augment.num_frequency_masks,
        max_frequency_mask_width=config.spec_augment.max_frequency_mask_width,
        num_time_masks=config.spec_augment.num_time_masks,
        max_time_mask_width=config.spec_augment.max_time_mask_width,
    )
    encoder = FastConformer(
        input_size=config.frontend.n_mels,
        hidden_size=config.model.encoder.hidden_size,
        num_heads=config.model.encoder.num_heads,
        kernel_size=config.model.encoder.kernel_size,
        num_blocks=config.model.encoder.num_blocks,
        dropout_rate=config.model.encoder.dropout_rate,
        conv_channels=config.model.encoder.conv_channels,
        feed_forward_expansion_factor=config.model.encoder.feed_forward_expansion_factor,
        bias=config.model.encoder.bias,
    )
    return FastConformerTransformer(
        frontend=frontend,
        spec_augment=spec_augment,
        encoder=encoder,
        vocab_size=config.model.vocab_size,
        blank_token_id=blank_token_id,
        ctc_loss_weight=config.model.ctc_loss_weight,
        decoder_hidden_size=config.model.decoder.hidden_size,
        decoder_num_layers=config.model.decoder.num_layers,
        decoder_num_heads=config.model.decoder.num_heads,
        decoder_feed_forward_size=config.model.decoder.feed_forward_size,
        decoder_dropout_rate=config.model.decoder.dropout_rate,
        decoder_max_length=config.model.decoder.max_length,
        label_smoothing=config.model.decoder.label_smoothing,
        ignore_index=config.model.decoder.ignore_index,
        bias=config.model.decoder.bias,
    )


def load_transformer_lm(
    model_dir: Path,
    tokenizer: PreTrainedTokenizerFast,
    device: torch.device,
) -> TransformerLM:
    """Load the Transformer LM and verify its tokenizer matches the ASR tokenizer."""
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
        raise ValueError("Language-model and ASR tokenizers must have the same vocabulary and token IDs")
    language_model_special_ids = (
        language_model_tokenizer.pad_token_id,
        language_model_tokenizer.bos_token_id,
        language_model_tokenizer.eos_token_id,
    )
    asr_special_ids = (tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id)
    if language_model_special_ids != asr_special_ids:
        raise ValueError("Language-model and ASR tokenizers must use the same PAD, BOS, and EOS token IDs")

    saved_config = cast(DictConfig, OmegaConf.load(config_path))
    model_config = saved_config.model
    if int(model_config.vocab_size) != len(tokenizer):
        raise ValueError("Language-model vocabulary size must match the ASR tokenizer")
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

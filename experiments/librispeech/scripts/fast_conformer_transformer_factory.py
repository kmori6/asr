from typing import cast

from omegaconf import DictConfig
from transformers import PreTrainedTokenizerFast

from asr.models import FastConformerTransformer
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

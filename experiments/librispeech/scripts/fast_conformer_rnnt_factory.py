from pathlib import Path
from typing import cast

import torch
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file
from transformers import PreTrainedTokenizerFast

from asr.models import FastConformerRNNT, StreamingFastConformerRNNT, TransformerLM
from asr.modules.conformer import FastConformer, StreamingFastConformer
from asr.modules.frontend import LogMelSpectrogram, SpecAugment
from asr.modules.rnnt import JointNetwork, PredictionNetwork

BLANK_TOKEN = "[BLANK]"


def load_model_weights(model: torch.nn.Module, model_path: Path, device: torch.device) -> None:
    """Load a Transformers Trainer directory, safetensors file, or legacy PyTorch weights."""
    weights_path = model_path / "model.safetensors" if model_path.is_dir() else model_path
    if not weights_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    if weights_path.suffix == ".safetensors":
        state_dict = load_file(str(weights_path), device=str(device))
    else:
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)


def validate_tokenizer(tokenizer: PreTrainedTokenizerFast, expected_vocab_size: int) -> tuple[int, int, int]:
    """Validate and return blank, BOS, and EOS token IDs."""
    actual_vocab_size = len(tokenizer)
    if actual_vocab_size != expected_vocab_size:
        raise ValueError(f"Tokenizer vocabulary size must be {expected_vocab_size}, but got {actual_vocab_size}.")

    vocabulary = tokenizer.get_vocab()
    if BLANK_TOKEN not in vocabulary:
        raise ValueError(f"Tokenizer must define {BLANK_TOKEN}.")
    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None or tokenizer.unk_token_id is None:
        raise ValueError("Tokenizer must define BOS, EOS, and UNK tokens.")

    tokenizer.pad_token = BLANK_TOKEN
    blank_token_id = cast(int, tokenizer.pad_token_id)
    bos_token_id = cast(int, tokenizer.bos_token_id)
    eos_token_id = cast(int, tokenizer.eos_token_id)
    if len({blank_token_id, bos_token_id, eos_token_id, cast(int, tokenizer.unk_token_id)}) != 4:
        raise ValueError("Blank, BOS, EOS, and UNK token IDs must be distinct.")
    return blank_token_id, bos_token_id, eos_token_id


def load_transformer_lm(
    model_dir: Path,
    tokenizer: PreTrainedTokenizerFast,
    device: torch.device,
) -> TransformerLM:
    """Load the Transformer LM and verify its tokenizer matches the RNN-T tokenizer."""
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
        raise ValueError("Language-model and RNN-T tokenizers must have the same vocabulary and token IDs")
    language_model_special_ids = (
        language_model_tokenizer.pad_token_id,
        language_model_tokenizer.bos_token_id,
        language_model_tokenizer.eos_token_id,
    )
    rnnt_special_ids = (tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id)
    if language_model_special_ids != rnnt_special_ids:
        raise ValueError("Language-model and RNN-T tokenizers must use the same PAD, BOS, and EOS token IDs")

    saved_config = cast(DictConfig, OmegaConf.load(config_path))
    model_config = saved_config.model
    if int(model_config.vocab_size) != len(tokenizer):
        raise ValueError("Language-model vocabulary size must match the RNN-T tokenizer")
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


def _build_frontend(config: DictConfig) -> LogMelSpectrogram:
    return LogMelSpectrogram(
        sample_rate=config.frontend.sample_rate,
        n_fft=config.frontend.n_fft,
        hop_length=config.frontend.hop_length,
        n_mels=config.frontend.n_mels,
        f_min=config.frontend.f_min,
        f_max=config.frontend.f_max,
    )


def _build_spec_augment(config: DictConfig) -> SpecAugment:
    return SpecAugment(
        num_frequency_masks=config.spec_augment.num_frequency_masks,
        max_frequency_mask_width=config.spec_augment.max_frequency_mask_width,
        num_time_masks=config.spec_augment.num_time_masks,
        max_time_mask_width=config.spec_augment.max_time_mask_width,
    )


def _build_prediction_network(config: DictConfig, blank_token_id: int) -> PredictionNetwork:
    return PredictionNetwork(
        vocab_size=config.model.vocab_size,
        hidden_size=config.model.prediction_network.hidden_size,
        num_layers=config.model.prediction_network.num_layers,
        dropout_rate=config.model.prediction_network.dropout_rate,
        blank_token_id=blank_token_id,
    )


def _build_joint_network(config: DictConfig) -> JointNetwork:
    return JointNetwork(
        vocab_size=config.model.vocab_size,
        encoder_size=config.model.encoder.hidden_size,
        predictor_size=config.model.prediction_network.hidden_size,
        hidden_size=config.model.joint_network.hidden_size,
        dropout_rate=config.model.joint_network.dropout_rate,
        bias=config.model.joint_network.bias,
    )


def build_fast_conformer_rnnt(config: DictConfig, blank_token_id: int) -> FastConformerRNNT:
    """Build the non-streaming LibriSpeech FastConformer RNN-T."""
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
    return FastConformerRNNT(
        frontend=_build_frontend(config),
        spec_augment=_build_spec_augment(config),
        encoder=encoder,
        prediction_network=_build_prediction_network(config, blank_token_id),
        joint_network=_build_joint_network(config),
        ctc_loss_weight=config.model.ctc_loss_weight,
        fastemit_lambda=config.model.fastemit_lambda,
    )


def build_streaming_fast_conformer_rnnt(config: DictConfig, blank_token_id: int) -> StreamingFastConformerRNNT:
    """Build the streaming LibriSpeech FastConformer RNN-T."""
    encoder = StreamingFastConformer(
        input_size=config.frontend.n_mels,
        hidden_size=config.model.encoder.hidden_size,
        num_heads=config.model.encoder.num_heads,
        kernel_size=config.model.encoder.kernel_size,
        num_blocks=config.model.encoder.num_blocks,
        dropout_rate=config.model.encoder.dropout_rate,
        min_chunk_size=config.model.encoder.min_chunk_size,
        max_chunk_size=config.model.encoder.max_chunk_size,
        streaming_mask_probability=config.model.encoder.streaming_mask_probability,
        conv_channels=config.model.encoder.conv_channels,
        feed_forward_expansion_factor=config.model.encoder.feed_forward_expansion_factor,
        bias=config.model.encoder.bias,
    )
    return StreamingFastConformerRNNT(
        frontend=_build_frontend(config),
        spec_augment=_build_spec_augment(config),
        encoder=encoder,
        prediction_network=_build_prediction_network(config, blank_token_id),
        joint_network=_build_joint_network(config),
        ctc_loss_weight=config.model.ctc_loss_weight,
        fastemit_lambda=config.model.fastemit_lambda,
    )

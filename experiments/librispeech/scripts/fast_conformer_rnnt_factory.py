from omegaconf import DictConfig
from transformers import PreTrainedTokenizerFast

from asr.models import FastConformerRNNT, StreamingFastConformerRNNT
from asr.modules.conformer import FastConformer, StreamingFastConformer
from asr.modules.frontend import LogMelSpectrogram, SpecAugment
from asr.modules.rnnt import JointNetwork, PredictionNetwork

BLANK_TOKEN = "[BLANK]"


def validate_tokenizer(tokenizer: PreTrainedTokenizerFast, expected_vocab_size: int) -> int:
    """Validate the tokenizer assumptions required by the RNN-T model."""
    actual_vocab_size = len(tokenizer)
    if actual_vocab_size != expected_vocab_size:
        raise ValueError(f"Tokenizer vocabulary size must be {expected_vocab_size}, but got {actual_vocab_size}.")

    vocabulary = tokenizer.get_vocab()
    if BLANK_TOKEN not in vocabulary:
        raise ValueError(f"Tokenizer must define {BLANK_TOKEN}.")
    if tokenizer.unk_token_id is None:
        raise ValueError("Tokenizer must define unk_token.")

    blank_token_id = vocabulary[BLANK_TOKEN]
    if blank_token_id == tokenizer.unk_token_id:
        raise ValueError("Blank and unknown tokens must have different IDs.")
    return blank_token_id


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

import math
from logging import getLogger
from pathlib import Path
from typing import cast

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from asr.data import SpeechTextCollator, SpeechTextDataset
from asr.models import FastConformerRNNT
from asr.modules.conformer import FastConformerEncoder
from asr.modules.frontend import LogMelSpectrogram, SpecAugment
from asr.modules.rnnt import JointNetwork, PredictionNetwork
from asr.training import LinearWarmupDecayLR, Trainer, TrainerArguments

logger = getLogger(__name__)
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
BLANK_TOKEN = "[BLANK]"


def resolve_experiment_path(path: str) -> Path:
    """Resolve a config path relative to the LibriSpeech experiment directory."""
    resolved_path = Path(path).expanduser()
    return resolved_path if resolved_path.is_absolute() else EXPERIMENT_DIR / resolved_path


def validate_tokenizer(tokenizer: PreTrainedTokenizerFast, expected_vocab_size: int) -> int:
    """Validate the tokenizer assumptions required for RNN-T training."""
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


def select_amp_dtype(device: torch.device) -> torch.dtype:
    """Use bfloat16 when supported, otherwise use float16 on CUDA."""
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        return torch.float16
    return torch.bfloat16


@hydra.main(version_base=None, config_path="../config", config_name="fast_conformer_rnnt")
def main(config: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.train.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.train.seed)
        torch.set_float32_matmul_precision("high")

    data_dir = resolve_experiment_path(config.dataset.data_dir)
    train_path = data_dir / config.dataset.train_file
    valid_path = data_dir / config.dataset.valid_file
    for data_path in (train_path, valid_path):
        if not data_path.is_file():
            raise FileNotFoundError(f"Dataset manifest not found: {data_path}")

    train_dataset = SpeechTextDataset(train_path, sample_rate=config.frontend.sample_rate)
    valid_dataset = SpeechTextDataset(valid_path, sample_rate=config.frontend.sample_rate)
    if len(train_dataset) == 0 or len(valid_dataset) == 0:
        raise ValueError("Training and validation datasets must not be empty.")

    tokenizer_dir = resolve_experiment_path(config.tokenizer.tokenizer_dir)
    if not tokenizer_dir.is_dir():
        raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_dir}")
    tokenizer = cast(
        PreTrainedTokenizerFast,
        PreTrainedTokenizerFast.from_pretrained(tokenizer_dir),
    )
    blank_token_id = validate_tokenizer(tokenizer, config.model.vocab_size)

    collate_fn = SpeechTextCollator(tokenizer, blank_token_id)
    num_workers = int(config.train.dataloader.num_workers)
    pin_memory = bool(config.train.dataloader.pin_memory) and device.type == "cuda"
    train_generator = torch.Generator().manual_seed(config.train.seed)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train.dataloader.train_batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config.train.dataloader.valid_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
    )

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
    encoder = FastConformerEncoder(
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
    prediction_network = PredictionNetwork(
        vocab_size=config.model.vocab_size,
        hidden_size=config.model.prediction_network.hidden_size,
        num_layers=config.model.prediction_network.num_layers,
        dropout_rate=config.model.prediction_network.dropout_rate,
        blank_token_id=blank_token_id,
    )
    joint_network = JointNetwork(
        vocab_size=config.model.vocab_size,
        encoder_size=config.model.encoder.hidden_size,
        predictor_size=config.model.prediction_network.hidden_size,
        hidden_size=config.model.joint_network.hidden_size,
        dropout_rate=config.model.joint_network.dropout_rate,
        bias=config.model.joint_network.bias,
    )
    model = FastConformerRNNT(
        frontend=frontend,
        spec_augment=spec_augment,
        encoder=encoder,
        prediction_network=prediction_network,
        joint_network=joint_network,
        ctc_loss_weight=config.model.ctc_loss_weight,
        fastemit_lambda=config.model.fastemit_lambda,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config.train.optimizer.lr,
        weight_decay=config.train.optimizer.weight_decay,
    )
    optimizer_steps_per_epoch = math.ceil(len(train_dataloader) / config.train.grad_accum_steps)
    scheduler = LinearWarmupDecayLR(
        optimizer,
        total_steps=optimizer_steps_per_epoch * config.train.epochs,
        warmup_steps=config.train.scheduler.warmup_steps,
    )

    amp_dtype = select_amp_dtype(device)
    scaler = GradScaler(device.type, enabled=device.type == "cuda" and amp_dtype == torch.float16)
    train_model: nn.Module = cast(nn.Module, torch.compile(model)) if config.train.compile else model
    out_dir = resolve_experiment_path(config.train.out_dir)
    checkpoint_path = (
        resolve_experiment_path(config.train.checkpoint_path) if config.train.checkpoint_path is not None else None
    )
    args = TrainerArguments(
        device=device,
        out_dir=out_dir,
        epochs=config.train.epochs,
        checkpoint_path=checkpoint_path,
        grad_accum_steps=config.train.grad_accum_steps,
        max_norm=config.train.optimizer.max_grad_norm,
        log_steps=config.train.log_steps,
        amp_dtype=amp_dtype,
    )
    trainer = Trainer(
        model=train_model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        args=args,
    )
    logger.info(
        "Starting training: device=%s, amp=%s, parameters=%d, optimizer_steps_per_epoch=%d",
        device,
        amp_dtype,
        sum(parameter.numel() for parameter in model.parameters()),
        optimizer_steps_per_epoch,
    )
    if checkpoint_path is not None:
        trainer.load_checkpoint()
    trainer.train()


if __name__ == "__main__":
    main()

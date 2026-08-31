from logging import getLogger
from pathlib import Path
from typing import cast

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file
from transformers import PreTrainedTokenizerFast, Trainer, TrainingArguments

from asr.data import CTCCollator, SpeechTextDataset
from asr.models import StreamingFastConformerCTC, TransformerLM
from asr.modules.conformer import StreamingFastConformer
from asr.modules.frontend import LogMelSpectrogram, SpecAugment

logger = getLogger(__name__)
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
BLANK_TOKEN = "[BLANK]"


def resolve_experiment_path(path: str) -> Path:
    """Resolve a config path relative to the LibriSpeech experiment directory."""
    resolved_path = Path(path).expanduser()
    return resolved_path if resolved_path.is_absolute() else EXPERIMENT_DIR / resolved_path


def validate_tokenizer(tokenizer: PreTrainedTokenizerFast, expected_vocab_size: int) -> tuple[int, int, int]:
    """Validate and return blank, BOS, and EOS token IDs."""
    if len(tokenizer) != expected_vocab_size:
        raise ValueError(f"Tokenizer vocabulary size must be {expected_vocab_size}, but got {len(tokenizer)}.")
    if BLANK_TOKEN not in tokenizer.get_vocab():
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


def build_streaming_fast_conformer_ctc(config: DictConfig, blank_token_id: int) -> StreamingFastConformerCTC:
    """Build the streaming FastConformer CTC experiment model."""
    frontend = LogMelSpectrogram(
        sample_rate=int(config.frontend.sample_rate),
        n_fft=int(config.frontend.n_fft),
        hop_length=int(config.frontend.hop_length),
        n_mels=int(config.frontend.n_mels),
        f_min=float(config.frontend.f_min),
        f_max=float(config.frontend.f_max),
    )
    spec_augment = SpecAugment(
        num_frequency_masks=int(config.spec_augment.num_frequency_masks),
        max_frequency_mask_width=int(config.spec_augment.max_frequency_mask_width),
        num_time_masks=int(config.spec_augment.num_time_masks),
        max_time_mask_width=int(config.spec_augment.max_time_mask_width),
    )
    encoder_config = config.model.encoder
    encoder = StreamingFastConformer(
        input_size=int(config.frontend.n_mels),
        hidden_size=int(encoder_config.hidden_size),
        num_heads=int(encoder_config.num_heads),
        kernel_size=int(encoder_config.kernel_size),
        num_blocks=int(encoder_config.num_blocks),
        dropout_rate=float(encoder_config.dropout_rate),
        min_chunk_size=int(encoder_config.min_chunk_size),
        max_chunk_size=int(encoder_config.max_chunk_size),
        streaming_mask_probability=float(encoder_config.streaming_mask_probability),
        conv_channels=int(encoder_config.conv_channels),
        feed_forward_expansion_factor=int(encoder_config.feed_forward_expansion_factor),
        bias=bool(encoder_config.bias),
    )
    return StreamingFastConformerCTC(
        frontend=frontend,
        spec_augment=spec_augment,
        encoder=encoder,
        vocab_size=int(config.model.vocab_size),
        blank_token_id=blank_token_id,
    )


def load_model_weights(model: torch.nn.Module, model_path: Path, device: torch.device) -> None:
    """Load Trainer safetensors or a PyTorch state dictionary."""
    weights_path = model_path / "model.safetensors" if model_path.is_dir() else model_path
    if not weights_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    if weights_path.suffix == ".safetensors":
        state_dict = load_file(str(weights_path), device=str(device))
    else:
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)


def load_transformer_lm(
    model_dir: Path,
    tokenizer: PreTrainedTokenizerFast,
    device: torch.device,
) -> TransformerLM:
    """Load a Transformer LM trained with the same tokenizer."""
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
    if (
        language_model_tokenizer.pad_token_id,
        language_model_tokenizer.bos_token_id,
        language_model_tokenizer.eos_token_id,
    ) != (tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id):
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


def select_mixed_precision(value: str, device: torch.device) -> tuple[bool, bool]:
    """Return the Transformers BF16 and FP16 flags for the requested policy."""
    if value == "auto":
        if device.type != "cuda":
            return False, False
        return (True, False) if torch.cuda.is_bf16_supported() else (False, True)
    if value == "fp32":
        return False, False
    if value == "bf16":
        if device.type != "cuda" or not torch.cuda.is_bf16_supported():
            raise ValueError("BF16 training requires a CUDA device with BF16 support.")
        return True, False
    if value == "fp16":
        if device.type != "cuda":
            raise ValueError("FP16 training requires a CUDA device.")
        return False, True
    raise ValueError("train.mixed_precision must be one of: auto, bf16, fp16, fp32")


@hydra.main(version_base=None, config_path="../config", config_name="streaming_fast_conformer_ctc")
def main(config: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    bf16, fp16 = select_mixed_precision(str(config.train.mixed_precision), device)

    data_dir = resolve_experiment_path(str(config.dataset.data_dir))
    train_path = data_dir / str(config.dataset.train_file)
    valid_path = data_dir / str(config.dataset.valid_file)
    for data_path in (train_path, valid_path):
        if not data_path.is_file():
            raise FileNotFoundError(f"Dataset manifest not found: {data_path}")

    sample_rate = int(config.frontend.sample_rate)
    train_dataset = SpeechTextDataset(train_path, sample_rate=sample_rate)
    valid_dataset = SpeechTextDataset(valid_path, sample_rate=sample_rate)
    if len(train_dataset) == 0 or len(valid_dataset) == 0:
        raise ValueError("Training and validation datasets must not be empty.")

    tokenizer_dir = resolve_experiment_path(str(config.tokenizer.tokenizer_dir))
    if not tokenizer_dir.is_dir():
        raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_dir}")
    tokenizer = cast(PreTrainedTokenizerFast, PreTrainedTokenizerFast.from_pretrained(tokenizer_dir))
    blank_token_id, _, _ = validate_tokenizer(tokenizer, int(config.model.vocab_size))
    model = build_streaming_fast_conformer_ctc(config, blank_token_id)
    collator = CTCCollator(tokenizer, blank_token_id)

    out_dir = resolve_experiment_path(str(config.train.out_dir))
    checkpoint_path = (
        resolve_experiment_path(str(config.train.checkpoint_path)) if config.train.checkpoint_path is not None else None
    )
    if checkpoint_path is not None and not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Transformers checkpoint directory not found: {checkpoint_path}")

    num_workers = int(config.train.dataloader_num_workers)
    training_arguments = TrainingArguments(
        output_dir=str(out_dir),
        do_train=True,
        do_eval=True,
        num_train_epochs=float(config.train.epochs),
        per_device_train_batch_size=int(config.train.train_batch_size),
        per_device_eval_batch_size=int(config.train.valid_batch_size),
        gradient_accumulation_steps=int(config.train.grad_accum_steps),
        learning_rate=float(config.train.learning_rate),
        weight_decay=float(config.train.weight_decay),
        max_grad_norm=float(config.train.max_grad_norm),
        lr_scheduler_type=str(config.train.lr_scheduler_type),
        warmup_steps=int(config.train.warmup_steps),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=int(config.train.logging_steps),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=int(config.train.save_total_limit),
        seed=int(config.train.seed),
        data_seed=int(config.train.seed),
        bf16=bf16,
        fp16=fp16,
        torch_compile=bool(config.train.compile),
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=bool(config.train.dataloader_pin_memory) and device.type == "cuda",
        dataloader_persistent_workers=num_workers > 0,
        remove_unused_columns=False,
        label_names=["labels", "label_lengths"],
        prediction_loss_only=True,
        optim="adamw_torch",
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_arguments,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
    )

    logger.info(
        "Starting streaming FastConformer CTC training: device=%s, bf16=%s, fp16=%s, parameters=%d",
        device,
        bf16,
        fp16,
        sum(parameter.numel() for parameter in model.parameters()),
    )
    train_result = trainer.train(resume_from_checkpoint=str(checkpoint_path) if checkpoint_path is not None else None)
    trainer.save_model()
    trainer.save_state()
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_metrics("eval", trainer.evaluate())


if __name__ == "__main__":
    main()

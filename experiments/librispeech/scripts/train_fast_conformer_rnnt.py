from logging import getLogger
from pathlib import Path
from typing import cast

import hydra
import torch
from fast_conformer_rnnt_factory import build_fast_conformer_rnnt, validate_tokenizer
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerFast, Trainer, TrainingArguments

from asr.data import RNNTCollator, SpeechTextDataset

logger = getLogger(__name__)
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]


def resolve_experiment_path(path: str) -> Path:
    """Resolve a config path relative to the LibriSpeech experiment directory."""
    resolved_path = Path(path).expanduser()
    return resolved_path if resolved_path.is_absolute() else EXPERIMENT_DIR / resolved_path


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


@hydra.main(version_base=None, config_path="../config", config_name="fast_conformer_rnnt")
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
    blank_token_id = validate_tokenizer(tokenizer, int(config.model.vocab_size))
    collator = RNNTCollator(tokenizer, blank_token_id)
    model = build_fast_conformer_rnnt(config, blank_token_id)

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
        "Starting FastConformer RNN-T training: device=%s, bf16=%s, fp16=%s, parameters=%d",
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

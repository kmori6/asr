from logging import getLogger
from pathlib import Path
from typing import cast

import hydra
import torch
from datasets import Dataset, load_dataset
from omegaconf import DictConfig, OmegaConf
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerFast, Trainer, TrainingArguments

from asr.models import TransformerLM

logger = getLogger(__name__)
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]


def resolve_experiment_path(path: str) -> Path:
    """Resolve a config path relative to the LibriSpeech experiment directory."""
    resolved_path = Path(path).expanduser()
    return resolved_path if resolved_path.is_absolute() else EXPERIMENT_DIR / resolved_path


def load_tokenizer(path: Path, config: DictConfig) -> PreTrainedTokenizerFast:
    """Load and validate the tokenizer shared by ASR and LM training."""
    if not path.is_dir():
        raise FileNotFoundError(f"Tokenizer directory not found: {path}")
    tokenizer = cast(PreTrainedTokenizerFast, PreTrainedTokenizerFast.from_pretrained(path))
    if len(tokenizer) != int(config.expected_vocab_size):
        raise ValueError(f"Tokenizer vocabulary size must be {config.expected_vocab_size}, but got {len(tokenizer)}.")

    vocabulary = tokenizer.get_vocab()
    required_tokens = [str(config.blank_token), str(config.bos_token), str(config.eos_token)]
    missing_tokens = [token for token in required_tokens if token not in vocabulary]
    if missing_tokens:
        raise ValueError(f"Tokenizer is missing required tokens: {missing_tokens}")
    if tokenizer.bos_token != config.bos_token or tokenizer.eos_token != config.eos_token:
        raise ValueError("Tokenizer BOS and EOS token roles do not match the LM configuration")

    tokenizer.pad_token = str(config.blank_token)
    return tokenizer


def load_tokenized_dataset(
    path: Path,
    tokenizer: PreTrainedTokenizerFast,
    max_length: int,
    num_proc: int,
) -> Dataset:
    """Load prepared LM JSON Lines and cache its tokenized Arrow representation."""
    if not path.is_file():
        raise FileNotFoundError(f"LM dataset not found: {path}")
    if max_length < 2:
        raise ValueError("max_length must be at least 2")
    if num_proc <= 0:
        raise ValueError("preprocessing_num_workers must be positive")
    with path.open("rb") as dataset_file:
        first_character = dataset_file.read(1024).lstrip()[:1]
    if first_character == b"[":
        raise ValueError(f"LM dataset must use JSON Lines; regenerate it with prepare_lm_dataset.py: {path}")

    dataset = cast(Dataset, load_dataset("json", data_files=str(path), split="train"))
    if "text" not in dataset.column_names:
        raise ValueError(f"LM dataset must contain a text field: {path}")

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        encoded = tokenizer(
            batch["text"],
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
        )
        return {
            "input_ids": cast(list[list[int]], encoded["input_ids"]),
            "attention_mask": cast(list[list[int]], encoded["attention_mask"]),
        }

    return dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=num_proc if num_proc > 1 else None,
        desc=f"Tokenizing {path.name}",
    )


def build_model(config: DictConfig) -> TransformerLM:
    """Build the Transformer LM described by the experiment configuration."""
    return TransformerLM(
        vocab_size=int(config.vocab_size),
        hidden_size=int(config.hidden_size),
        num_heads=int(config.num_heads),
        num_layers=int(config.num_layers),
        feed_forward_size=int(config.feed_forward_size),
        dropout_rate=float(config.dropout_rate),
        max_length=int(config.max_length),
        bias=bool(config.bias),
    )


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


@hydra.main(version_base=None, config_path="../config", config_name="transformer_lm")
def main(config: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    bf16, fp16 = select_mixed_precision(str(config.train.mixed_precision), device)

    tokenizer_dir = resolve_experiment_path(str(config.tokenizer.tokenizer_dir))
    tokenizer = load_tokenizer(tokenizer_dir, config.tokenizer)
    if len(tokenizer) != int(config.model.vocab_size):
        raise ValueError(f"Model vocab_size must match tokenizer size: {config.model.vocab_size} != {len(tokenizer)}")

    data_dir = resolve_experiment_path(str(config.dataset.data_dir))
    num_proc = int(config.dataset.preprocessing_num_workers)
    train_dataset = load_tokenized_dataset(
        data_dir / str(config.dataset.train_file), tokenizer, int(config.model.max_length), num_proc
    )
    valid_dataset = load_tokenized_dataset(
        data_dir / str(config.dataset.valid_file), tokenizer, int(config.model.max_length), num_proc
    )
    if len(train_dataset) == 0 or len(valid_dataset) == 0:
        raise ValueError("Training and validation datasets must not be empty")

    model = build_model(config.model)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    out_dir = resolve_experiment_path(str(config.train.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, out_dir / "config.yaml", resolve=True)
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
        "Starting Transformer LM training: device=%s, bf16=%s, fp16=%s, parameters=%d",
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

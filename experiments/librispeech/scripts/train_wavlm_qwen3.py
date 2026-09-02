from logging import getLogger
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, Trainer, TrainingArguments, Wav2Vec2FeatureExtractor
from transformers.trainer import TRAINING_ARGS_NAME, WEIGHTS_NAME

from asr.data import SpeechTextDataset, WavlmQwen3Collator
from asr.models import WavLMQwen3

logger = getLogger(__name__)
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]


class WavLMQwen3Trainer(Trainer):
    """Save the composite model with PyTorch to preserve Qwen3's tied weights."""

    def _save(self, output_dir: str | None = None, state_dict: dict[str, torch.Tensor] | None = None) -> None:
        resolved_output_dir = output_dir if output_dir is not None else self.args.output_dir
        if resolved_output_dir is None:
            raise RuntimeError("Trainer output directory is required when saving a checkpoint")
        output_path = Path(resolved_output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        if self.model is None:
            raise RuntimeError("Trainer model is required when saving a checkpoint")
        torch.save(self.model.state_dict() if state_dict is None else state_dict, output_path / WEIGHTS_NAME)
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_path)
        torch.save(self.args, output_path / TRAINING_ARGS_NAME)


def resolve_experiment_path(path: str) -> Path:
    """Resolve a config path relative to the LibriSpeech experiment directory."""
    resolved_path = Path(path).expanduser()
    return resolved_path if resolved_path.is_absolute() else EXPERIMENT_DIR / resolved_path


def select_mixed_precision(value: str, device: torch.device) -> tuple[bool, bool, torch.dtype]:
    """Return BF16, FP16, and model dtype for the requested precision policy."""
    if value == "auto":
        if device.type != "cuda":
            return False, False, torch.float32
        return (True, False, torch.bfloat16) if torch.cuda.is_bf16_supported() else (False, True, torch.float16)
    if value == "fp32":
        return False, False, torch.float32
    if value == "bf16":
        if device.type != "cuda" or not torch.cuda.is_bf16_supported():
            raise ValueError("BF16 training requires a CUDA device with BF16 support.")
        return True, False, torch.bfloat16
    if value == "fp16":
        if device.type != "cuda":
            raise ValueError("FP16 training requires a CUDA device.")
        return False, True, torch.float16
    raise ValueError("train.mixed_precision must be one of: auto, bf16, fp16, fp32")


@hydra.main(version_base=None, config_path="../config", config_name="wavlm_qwen3")
def main(config: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    bf16, fp16, model_dtype = select_mixed_precision(str(config.train.mixed_precision), device)

    data_dir = resolve_experiment_path(str(config.dataset.data_dir))
    train_path = data_dir / str(config.dataset.train_file)
    valid_path = data_dir / str(config.dataset.valid_file)
    for data_path in (train_path, valid_path):
        if not data_path.is_file():
            raise FileNotFoundError(f"Dataset manifest not found: {data_path}")

    sample_rate = int(config.dataset.sample_rate)
    train_dataset = SpeechTextDataset(train_path, sample_rate=sample_rate)
    valid_dataset = SpeechTextDataset(valid_path, sample_rate=sample_rate)
    if len(train_dataset) == 0 or len(valid_dataset) == 0:
        raise ValueError("Training and validation datasets must not be empty.")

    speech_encoder_name = str(config.model.speech_encoder_name_or_path)
    language_model_name = str(config.model.language_model_name_or_path)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(speech_encoder_name)
    tokenizer = AutoTokenizer.from_pretrained(language_model_name)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("Qwen3 tokenizer must define an EOS token that can be used for padding.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    collator = WavlmQwen3Collator(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        sample_rate=sample_rate,
        language=str(config.model.language),
        max_text_length=int(config.model.max_text_length),
    )

    model = WavLMQwen3.from_pretrained(
        speech_encoder_name_or_path=speech_encoder_name,
        language_model_name_or_path=language_model_name,
        audio_downsample_factor=int(config.model.audio_downsample_factor),
        dtype=model_dtype,
    )
    if bool(config.model.freeze_feature_encoder):
        model.freeze_feature_encoder()

    llm_adaptation = str(config.model.llm.adaptation)
    if llm_adaptation == "lora":
        model.add_language_model_lora(
            rank=int(config.model.llm.lora.rank),
            alpha=int(config.model.llm.lora.alpha),
            dropout=float(config.model.llm.lora.dropout),
            target_modules=[str(name) for name in config.model.llm.lora.target_modules],
        )
    elif llm_adaptation == "frozen":
        model.language_model.requires_grad_(False)
    else:
        raise ValueError("model.llm.adaptation must be one of: lora, frozen")
    model.language_model.config.use_cache = False

    weight_decay = float(config.train.weight_decay)
    learning_rate = float(config.train.learning_rate)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

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
        learning_rate=learning_rate,
        weight_decay=weight_decay,
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
        gradient_checkpointing=bool(config.train.gradient_checkpointing),
        torch_compile=bool(config.train.compile),
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=bool(config.train.dataloader_pin_memory) and device.type == "cuda",
        dataloader_persistent_workers=num_workers > 0,
        remove_unused_columns=False,
        prediction_loss_only=True,
        label_names=["labels"],
        report_to="none",
    )
    trainer = WavLMQwen3Trainer(
        model=model,
        args=training_arguments,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
        optimizers=(optimizer, None),
    )

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    logger.info(
        "Starting WavLM-Qwen3 training: device=%s, bf16=%s, fp16=%s, trainable=%d/%d",
        device,
        bf16,
        fp16,
        trainable_parameters,
        total_parameters,
    )
    train_result = trainer.train(resume_from_checkpoint=str(checkpoint_path) if checkpoint_path is not None else None)
    model.language_model.config.use_cache = True
    trainer.save_model()
    tokenizer.save_pretrained(out_dir / "tokenizer")
    feature_extractor.save_pretrained(out_dir / "feature_extractor")
    resolved_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    OmegaConf.save(resolved_config, out_dir / "resolved_config.yaml")
    trainer.save_state()
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_metrics("eval", trainer.evaluate())


if __name__ == "__main__":
    main()

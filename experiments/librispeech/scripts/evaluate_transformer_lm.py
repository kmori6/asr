import json
import math
from logging import getLogger
from typing import cast

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file
from train_transformer_lm import (
    build_model,
    load_tokenized_dataset,
    load_tokenizer,
    resolve_experiment_path,
    select_mixed_precision,
)
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

logger = getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="transformer_lm")
def main(config: DictConfig) -> None:
    model_path = resolve_experiment_path(str(config.evaluate.model_path))
    weights_path = model_path / "model.safetensors"
    saved_config_path = model_path / "config.yaml"
    if not weights_path.is_file():
        raise FileNotFoundError(f"Transformer LM weights not found: {weights_path}")
    if not saved_config_path.is_file():
        raise FileNotFoundError(f"Transformer LM configuration not found: {saved_config_path}")
    saved_config = cast(DictConfig, OmegaConf.load(saved_config_path))

    tokenizer = load_tokenizer(model_path, saved_config.tokenizer)
    data_dir = resolve_experiment_path(str(config.dataset.data_dir))
    dataset = load_tokenized_dataset(
        data_dir / str(config.dataset.valid_file),
        tokenizer,
        int(saved_config.model.max_length),
        int(config.dataset.preprocessing_num_workers),
    )
    max_samples = config.evaluate.max_samples
    if max_samples is not None:
        if int(max_samples) <= 0:
            raise ValueError("evaluate.max_samples must be positive or null")
        dataset = dataset.select(range(min(len(dataset), int(max_samples))))
    if len(dataset) == 0:
        raise ValueError("The evaluation dataset must not be empty")

    model = build_model(saved_config.model)
    model.load_state_dict(load_file(weights_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    bf16, fp16 = select_mixed_precision(str(config.evaluate.mixed_precision), device)
    out_dir = resolve_experiment_path(str(config.evaluate.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out_dir),
            do_eval=True,
            per_device_eval_batch_size=int(config.evaluate.batch_size),
            bf16=bf16,
            fp16=fp16,
            dataloader_num_workers=int(config.evaluate.dataloader_num_workers),
            dataloader_pin_memory=device.type == "cuda",
            prediction_loss_only=True,
            report_to="none",
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        eval_dataset=dataset,
        processing_class=tokenizer,
    )
    metrics = trainer.evaluate()
    eval_loss = float(metrics["eval_loss"])
    try:
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = math.inf
    metrics["perplexity"] = perplexity
    metrics["num_samples"] = len(dataset)

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, ensure_ascii=False, indent=2)
        metrics_file.write("\n")
    logger.info("Validation loss: %.4f, perplexity: %.4f", eval_loss, perplexity)


if __name__ == "__main__":
    main()

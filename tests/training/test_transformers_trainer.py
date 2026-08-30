import math
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from asr.models import FastConformerRNNT, StreamingFastConformerRNNT
from asr.modules.conformer import FastConformer, StreamingFastConformer
from asr.modules.frontend import LogMelSpectrogram, SpecAugment
from asr.modules.rnnt import JointNetwork, PredictionNetwork


class _RNNTDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self) -> None:
        self.sample = {
            "waveforms": torch.randn(256),
            "waveform_lengths": torch.tensor(256),
            "labels": torch.tensor([2, 3]),
            "label_lengths": torch.tensor(2),
        }

    def __len__(self) -> int:
        return 1

    def __getitem__(self, _index: int) -> dict[str, torch.Tensor]:
        return self.sample


def _collate(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {name: torch.stack([sample[name] for sample in samples]) for name in samples[0]}


def _create_model(streaming: bool) -> FastConformerRNNT:
    encoder: FastConformer
    model_type: type[FastConformerRNNT]
    if streaming:
        encoder = StreamingFastConformer(
            input_size=4,
            hidden_size=8,
            num_heads=2,
            kernel_size=5,
            num_blocks=1,
            dropout_rate=0.0,
            min_chunk_size=2,
            max_chunk_size=2,
            streaming_mask_probability=1.0,
            conv_channels=4,
        )
        model_type = StreamingFastConformerRNNT
    else:
        encoder = FastConformer(
            input_size=4,
            hidden_size=8,
            num_heads=2,
            kernel_size=5,
            num_blocks=1,
            dropout_rate=0.0,
            conv_channels=4,
        )
        model_type = FastConformerRNNT

    return model_type(
        frontend=LogMelSpectrogram(n_fft=16, hop_length=8, n_mels=4),
        spec_augment=SpecAugment(0, 0, 0, 0),
        encoder=encoder,
        prediction_network=PredictionNetwork(
            vocab_size=8,
            hidden_size=6,
            num_layers=1,
            dropout_rate=0.0,
            blank_token_id=0,
        ),
        joint_network=JointNetwork(
            vocab_size=8,
            encoder_size=8,
            predictor_size=6,
            hidden_size=10,
            dropout_rate=0.0,
        ),
        ctc_loss_weight=0.3,
        fastemit_lambda=0.004,
    )


def _run_trainer_smoke(tmp_path: Path, streaming: bool) -> None:
    dataset = _RNNTDataset()
    output_dir = tmp_path / ("streaming" if streaming else "full_context")
    trainer = Trainer(
        model=_create_model(streaming),
        args=TrainingArguments(
            output_dir=str(output_dir),
            max_steps=1,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            learning_rate=1e-4,
            eval_strategy="no",
            save_strategy="no",
            logging_strategy="no",
            prediction_loss_only=True,
            remove_unused_columns=False,
            label_names=["labels", "label_lengths"],
            optim="adamw_torch",
            report_to="none",
            use_cpu=True,
        ),
        data_collator=_collate,
        train_dataset=dataset,
        eval_dataset=dataset,
    )

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    trainer.save_model()

    assert train_result.global_step == 1
    assert math.isfinite(train_result.training_loss)
    assert math.isfinite(eval_metrics["eval_loss"])
    assert (output_dir / "model.safetensors").is_file()


def test_transformers_trainer_accepts_full_context_rnnt_loss_mapping(tmp_path: Path) -> None:
    _run_trainer_smoke(tmp_path, streaming=False)


def test_transformers_trainer_accepts_streaming_rnnt_loss_mapping(tmp_path: Path) -> None:
    _run_trainer_smoke(tmp_path, streaming=True)

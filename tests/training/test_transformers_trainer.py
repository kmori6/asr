import math
import runpy
from pathlib import Path
from typing import cast

import torch
from torch.utils.data import Dataset
from transformers import Qwen3Config, Qwen3ForCausalLM, Trainer, TrainingArguments, WavLMConfig, WavLMModel

from asr.models import FastConformerRNNT, StreamingFastConformerCTC, StreamingFastConformerRNNT, WavLMQwen3
from asr.modules.conformer import FastConformer, StreamingFastConformer
from asr.modules.frontend import LogMelSpectrogram, SpecAugment
from asr.modules.rnnt import JointNetwork, PredictionNetwork


class _ASRDataset(Dataset[dict[str, torch.Tensor]]):
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


class _EncoderLLMDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self) -> None:
        self.sample = {
            "waveforms": torch.randn(80),
            "waveform_lengths": torch.tensor(80),
            "input_ids": torch.tensor([1, 5, 2]),
            "attention_mask": torch.ones(3, dtype=torch.bool),
            "labels": torch.tensor([-100, 5, 2]),
        }

    def __len__(self) -> int:
        return 1

    def __getitem__(self, _index: int) -> dict[str, torch.Tensor]:
        return self.sample


def _collate(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {name: torch.stack([sample[name] for sample in samples]) for name in samples[0]}


def _create_rnnt_model(streaming: bool) -> FastConformerRNNT:
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


def _create_streaming_ctc_model() -> StreamingFastConformerCTC:
    return StreamingFastConformerCTC(
        frontend=LogMelSpectrogram(n_fft=16, hop_length=8, n_mels=4),
        spec_augment=SpecAugment(0, 0, 0, 0),
        encoder=StreamingFastConformer(
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
        ),
        vocab_size=8,
        blank_token_id=0,
    )


def _create_wavlm_qwen3_model() -> WavLMQwen3:
    speech_encoder = WavLMModel(
        WavLMConfig(
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            conv_dim=(8, 8),
            conv_stride=(2, 2),
            conv_kernel=(4, 2),
            conv_bias=False,
            num_conv_pos_embeddings=4,
            num_conv_pos_embedding_groups=2,
            mask_time_prob=0.0,
        )
    )
    language_model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            max_position_embeddings=64,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=True,
        )
    )
    model = WavLMQwen3(speech_encoder, language_model, audio_downsample_factor=2)
    model.freeze_feature_encoder()
    model.add_language_model_lora(
        rank=2,
        alpha=4,
        dropout=0.0,
        target_modules=["q_proj", "v_proj"],
    )
    return model


def _run_trainer_smoke(
    tmp_path: Path,
    model: FastConformerRNNT | StreamingFastConformerCTC,
    output_name: str,
) -> None:
    dataset = _ASRDataset()
    output_dir = tmp_path / output_name
    trainer = Trainer(
        model=model,
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
    _run_trainer_smoke(tmp_path, _create_rnnt_model(streaming=False), "full_context_rnnt")


def test_transformers_trainer_accepts_streaming_rnnt_loss_mapping(tmp_path: Path) -> None:
    _run_trainer_smoke(tmp_path, _create_rnnt_model(streaming=True), "streaming_rnnt")


def test_transformers_trainer_accepts_streaming_ctc_loss_mapping(tmp_path: Path) -> None:
    _run_trainer_smoke(tmp_path, _create_streaming_ctc_model(), "streaming_ctc")


def test_transformers_trainer_saves_wavlm_qwen3_with_lora(tmp_path: Path) -> None:
    dataset = _EncoderLLMDataset()
    output_dir = tmp_path / "wavlm_qwen3"
    script_path = Path(__file__).parents[2] / "experiments/librispeech/scripts/train_wavlm_qwen3.py"
    trainer_class = cast(type[Trainer], runpy.run_path(str(script_path))["WavLMQwen3Trainer"])
    trainer = trainer_class(
        model=_create_wavlm_qwen3_model(),
        args=TrainingArguments(
            output_dir=str(output_dir),
            max_steps=1,
            per_device_train_batch_size=1,
            eval_strategy="no",
            save_strategy="no",
            logging_strategy="no",
            remove_unused_columns=False,
            label_names=["labels"],
            report_to="none",
            use_cpu=True,
        ),
        data_collator=_collate,
        train_dataset=dataset,
    )

    train_result = trainer.train()
    trainer.save_model()

    assert train_result.global_step == 1
    weights_path = output_dir / "pytorch_model.bin"
    assert weights_path.is_file()
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    _create_wavlm_qwen3_model().load_state_dict(state_dict)

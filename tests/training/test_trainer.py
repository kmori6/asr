from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader, Dataset

from asr.training.scheduler import LinearWarmupDecayLR
from asr.training.trainer import Trainer, TrainerArguments


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        prediction = self.weight * x
        return {
            "loss": prediction.square().mean(),
            "acc": (prediction > 0).float().mean(),
            "logits": torch.stack([prediction, -prediction], dim=-1),
        }


class _ToyDataset(Dataset[dict[str, torch.Tensor]]):
    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"x": torch.tensor(float(index + 1))}


def test_trainer_flushes_partial_accumulation_and_saves_compiled_model(tmp_path: Path) -> None:
    dataloader = DataLoader(_ToyDataset(), batch_size=1)
    raw_model = _ToyModel()
    optimizer = torch.optim.SGD(raw_model.parameters(), lr=1.0)
    scheduler = LinearWarmupDecayLR(optimizer, total_steps=4, warmup_steps=1)
    compiled_model = cast(nn.Module, torch.compile(raw_model, backend="eager"))
    trainer = Trainer(
        model=compiled_model,
        train_dataloader=dataloader,
        valid_dataloader=dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=GradScaler("cpu", enabled=False),
        args=TrainerArguments(
            device=torch.device("cpu"),
            out_dir=tmp_path,
            epochs=1,
            grad_accum_steps=2,
        ),
    )

    train_metrics = trainer.train_epoch()
    valid_metrics = trainer.validate_epoch()
    trainer.save_checkpoint(epoch=1)

    assert scheduler.last_epoch == 2
    assert raw_model.weight.grad is None
    assert torch.isfinite(torch.tensor([train_metrics["loss"], valid_metrics["loss"]])).all()
    assert train_metrics.keys() == {"loss", "acc"}
    assert valid_metrics.keys() == {"loss", "acc"}
    checkpoint = torch.load(tmp_path / "checkpoint_epoch_1.pt", weights_only=True)
    assert checkpoint["model_state_dict"].keys() == {"weight"}

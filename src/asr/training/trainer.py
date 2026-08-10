from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from asr.training.history import TrainingHistory


@dataclass
class TrainerArguments:
    device: torch.device
    out_dir: Path
    epochs: int
    checkpoint_path: Path | None = None
    grad_accum_steps: int = 1
    max_norm: float = 1.0
    log_steps: int = 10
    amp_dtype: torch.dtype = torch.bfloat16


class Trainer:
    """Trainer for training a model.

    Tuning guide: https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html

    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        scaler: GradScaler,
        args: TrainerArguments,
    ) -> None:
        if args.epochs <= 0:
            raise ValueError(f"epochs must be positive, but got {args.epochs}.")
        if args.grad_accum_steps <= 0:
            raise ValueError(f"grad_accum_steps must be positive, but got {args.grad_accum_steps}.")
        if args.max_norm <= 0:
            raise ValueError(f"max_norm must be positive, but got {args.max_norm}.")
        if args.log_steps <= 0:
            raise ValueError(f"log_steps must be positive, but got {args.log_steps}.")

        self.args = args
        self.model = model.to(self.args.device)
        self.checkpoint_model = cast(nn.Module, getattr(self.model, "_orig_mod", self.model))
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.history = TrainingHistory(self.args.out_dir)
        self.start_epoch = 0
        self.best_valid_loss = float("inf")

    def _forward_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        with autocast(self.args.device.type, dtype=self.args.amp_dtype):
            outputs = cast(
                dict[str, torch.Tensor],
                self.model(**{name: tensor.to(self.args.device, non_blocking=True) for name, tensor in batch.items()}),
            )
        if "loss" not in outputs:
            raise ValueError("Model must return a loss")
        return outputs

    def train_epoch(self) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        num_batches = len(self.train_dataloader)
        if num_batches == 0:
            raise ValueError("train_dataloader must not be empty.")

        metric_sums: dict[str, torch.Tensor] = {}
        metric_counts: dict[str, int] = {}
        for step, batch in tqdm(enumerate(self.train_dataloader, 1), total=num_batches, dynamic_ncols=True):
            outputs = self._forward_batch(batch)
            window_start = ((step - 1) // self.args.grad_accum_steps) * self.args.grad_accum_steps
            window_size = min(self.args.grad_accum_steps, num_batches - window_start)
            loss = outputs["loss"] / window_size
            self.scaler.scale(loss).backward()
            should_step = step % self.args.grad_accum_steps == 0 or step == num_batches
            if should_step:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.max_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
            if step % self.args.log_steps == 0:
                msg = f"lr: {self.scheduler.get_last_lr()[0]:.6f}, "
                for name, value in outputs.items():
                    if value.numel() == 1:
                        msg += f"{name}: {value.item():.3f}, "
                tqdm.write(msg)
            for name, value in outputs.items():
                if value.numel() != 1:
                    continue
                if name in metric_sums:
                    metric_sums[name].add_(value.detach().float())
                    metric_counts[name] += 1
                else:
                    metric_sums[name] = value.detach().float().clone()
                    metric_counts[name] = 1

        # NOTE: Epoch metrics are unweighted averages of per-batch metrics. They are intended for
        # monitoring and may differ slightly from exact token-level averages.
        return {name: (total / metric_counts[name]).item() for name, total in metric_sums.items()}

    @torch.inference_mode()
    def validate_epoch(self) -> dict[str, float]:
        self.model.eval()
        num_batches = len(self.valid_dataloader)
        if num_batches == 0:
            raise ValueError("valid_dataloader must not be empty.")

        metric_sums: dict[str, torch.Tensor] = {}
        metric_counts: dict[str, int] = {}
        for batch in tqdm(
            self.valid_dataloader,
            desc="validating",
            total=num_batches,
            leave=False,
            dynamic_ncols=True,
        ):
            outputs = self._forward_batch(batch)
            for name, value in outputs.items():
                if value.numel() != 1:
                    continue
                if name in metric_sums:
                    metric_sums[name].add_(value.detach().float())
                    metric_counts[name] += 1
                else:
                    metric_sums[name] = value.detach().float().clone()
                    metric_counts[name] = 1

        # NOTE: See train_epoch(); these are unweighted per-batch averages for monitoring.
        return {name: (total / metric_counts[name]).item() for name, total in metric_sums.items()}

    def save_checkpoint(self, epoch: int) -> None:
        state = {
            "epoch": epoch,
            "model_state_dict": self.checkpoint_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_valid_loss": self.best_valid_loss,
        }
        torch.save(state, self.args.out_dir / f"checkpoint_epoch_{epoch}.pt")
        # Remove old checkpoints, keep only the latest
        for old_ckpt in self.args.out_dir.glob("checkpoint_epoch_*.pt"):
            if old_ckpt.name != f"checkpoint_epoch_{epoch}.pt":
                old_ckpt.unlink()

    def load_checkpoint(self) -> None:
        if self.args.checkpoint_path is None:
            raise ValueError("args.checkpoint_path must be set to load a checkpoint")
        checkpoint = torch.load(
            self.args.checkpoint_path,
            map_location=self.args.device,
            weights_only=True,
        )
        self.checkpoint_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.best_valid_loss = checkpoint["best_valid_loss"]
        self.start_epoch = checkpoint["epoch"]

    def train(self) -> None:
        self.args.out_dir.mkdir(parents=True, exist_ok=True)
        for epoch in tqdm(range(self.start_epoch, self.args.epochs), desc="Training", dynamic_ncols=True):
            train_metrics = self.train_epoch()
            valid_metrics = self.validate_epoch()
            train_summary = ", ".join(f"{name}: {value:.3f}" for name, value in train_metrics.items())
            valid_summary = ", ".join(f"{name}: {value:.3f}" for name, value in valid_metrics.items())
            tqdm.write(f"epoch {epoch + 1}/{self.args.epochs}, train [{train_summary}], valid [{valid_summary}]")

            valid_loss = valid_metrics["loss"]
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                torch.save(self.checkpoint_model.state_dict(), self.args.out_dir / "best_model.pt")
            self.history.append(epoch + 1, train_metrics, valid_metrics)
            self.history.save()
            self.save_checkpoint(epoch + 1)

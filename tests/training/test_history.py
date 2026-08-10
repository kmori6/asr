import json
from pathlib import Path

from asr.training.history import TrainingHistory


def test_training_history_saves_and_restores_generic_metrics(tmp_path: Path) -> None:
    history = TrainingHistory(tmp_path)
    history.append(
        epoch=1,
        train_metrics={"loss": 3.0, "acc": 0.4, "wer": 0.2},
        valid_metrics={"loss": 2.5, "acc": 0.5, "wer": 0.15},
    )
    history.save()

    with (tmp_path / "history.json").open(encoding="utf-8") as file:
        records = json.load(file)
    assert records == [
        {
            "epoch": 1,
            "train_loss": 3.0,
            "train_acc": 0.4,
            "train_wer": 0.2,
            "valid_loss": 2.5,
            "valid_acc": 0.5,
            "valid_wer": 0.15,
        }
    ]
    assert all((tmp_path / name).stat().st_size > 0 for name in ("loss.png", "acc.png", "wer.png"))

    restored = TrainingHistory(tmp_path)
    assert restored.records == records

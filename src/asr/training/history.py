import json
from pathlib import Path
from typing import cast

from matplotlib.figure import Figure


class TrainingHistory:
    """Persist epoch metrics as JSON and line plots."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.history_path = out_dir / "history.json"
        self.records: list[dict[str, int | float]]
        if self.history_path.exists():
            with self.history_path.open(encoding="utf-8") as file:
                self.records = cast(list[dict[str, int | float]], json.load(file))
        else:
            self.records = []

    def append(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        valid_metrics: dict[str, float],
    ) -> None:
        """Add or replace metrics for one epoch."""
        record: dict[str, int | float] = {"epoch": epoch}
        record.update({f"train_{name}": value for name, value in train_metrics.items()})
        record.update({f"valid_{name}": value for name, value in valid_metrics.items()})
        self.records = [item for item in self.records if item["epoch"] != epoch]
        self.records.append(record)
        self.records.sort(key=lambda item: item["epoch"])

    def save(self) -> None:
        """Write the history JSON and regenerate plots for every metric."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = self.history_path.with_name(f"{self.history_path.name}.tmp")
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(self.records, file, indent=2, ensure_ascii=False)
            file.write("\n")
        temporary_path.replace(self.history_path)

        metric_names = sorted(
            {
                key.removeprefix(prefix)
                for record in self.records
                for key in record
                for prefix in ("train_", "valid_")
                if key.startswith(prefix)
            }
        )
        for metric_name in metric_names:
            self._save_plot(metric_name)

    def _save_plot(self, metric_name: str) -> None:
        figure = Figure(figsize=(8, 5))
        axis = figure.subplots()
        for split in ("train", "valid"):
            key = f"{split}_{metric_name}"
            points = [(int(record["epoch"]), float(record[key])) for record in self.records if key in record]
            if points:
                epochs, values = zip(*points)
                axis.plot(epochs, values, marker="o", label=split)

        axis.set_xlabel("Epoch")
        axis.set_ylabel(metric_name)
        axis.set_title(metric_name)
        axis.grid(alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(self.out_dir / f"{metric_name.replace('/', '_')}.png")
        figure.clear()

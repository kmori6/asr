import torch

from asr.training.scheduler import LinearWarmupDecayLR


def test_linear_warmup_decay_lr_schedule() -> None:
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = LinearWarmupDecayLR(
        optimizer,
        total_steps=4,
        warmup_steps=2,
    )

    learning_rates = [float(optimizer.param_groups[0]["lr"])]
    for _ in range(5):
        optimizer.step()
        scheduler.step()
        learning_rates.append(float(optimizer.param_groups[0]["lr"]))

    torch.testing.assert_close(
        torch.tensor(learning_rates),
        torch.tensor([0.0, 0.5, 1.0, 0.5, 0.0, 0.0]),
    )

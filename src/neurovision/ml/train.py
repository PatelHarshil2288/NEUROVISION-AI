from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from neurovision.ml.data import flatten_batch


@dataclass(frozen=True)
class TrainProgress:
    epoch: int
    step: int
    steps_per_epoch: int
    train_loss: float
    test_accuracy: float | None


def evaluate_accuracy(model: nn.Module, test_dl: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_dl:
            x = flatten_batch(x.to(device))
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    return float(correct) / float(max(1, total))


def train_epochs(
    *,
    model: nn.Module,
    train_dl: DataLoader,
    test_dl: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    on_progress: Optional[Callable[[TrainProgress], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> None:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        steps_per_epoch = len(train_dl)

        for step, (x, y) in enumerate(train_dl, start=1):
            if should_stop is not None and should_stop():
                return

            x = flatten_batch(x.to(device))
            y = y.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            global_step += 1

            if on_progress is not None and (step % 50 == 0 or step == steps_per_epoch):
                acc = None
                if test_dl is not None and (step == steps_per_epoch):
                    acc = evaluate_accuracy(model, test_dl, device)

                on_progress(
                    TrainProgress(
                        epoch=epoch,
                        step=step,
                        steps_per_epoch=steps_per_epoch,
                        train_loss=float(loss.item()),
                        test_accuracy=acc,
                    )
                )

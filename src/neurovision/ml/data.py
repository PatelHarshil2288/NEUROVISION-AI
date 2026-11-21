from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def mnist_dataloaders(
    *,
    data_dir: str | Path,
    batch_size: int = 128,
    num_workers: int = 0,
    download: bool = True,
) -> tuple[DataLoader, DataLoader]:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    train_ds = datasets.MNIST(root=str(data_dir), train=True, transform=tfm, download=download)
    test_ds = datasets.MNIST(root=str(data_dir), train=False, transform=tfm, download=download)

    pin = False
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    return train_dl, test_dl


def flatten_batch(x: torch.Tensor) -> torch.Tensor:
    # x: (N,1,28,28) -> (N,784)
    return x.view(x.shape[0], -1)

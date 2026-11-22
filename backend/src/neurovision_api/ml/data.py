from __future__ import annotations

from collections.abc import Sized
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

MNIST_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ]
)


@dataclass(frozen=True)
class MNISTDataLoaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader


def split_training_dataset(
    training_dataset: Dataset,
    *,
    validation_size: int = 5_000,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    if not isinstance(training_dataset, Sized):
        raise TypeError("The training dataset must have a defined length.")
    dataset_size = len(training_dataset)
    if validation_size < 1 or validation_size >= dataset_size:
        raise ValueError(
            "validation_size must be greater than zero and smaller than "
            f"the training dataset size ({dataset_size})."
        )

    training_size = dataset_size - validation_size
    generator = torch.Generator().manual_seed(seed)
    training_subset, validation_subset = random_split(
        training_dataset,
        [training_size, validation_size],
        generator=generator,
    )
    return training_subset, validation_subset


def make_mnist_loaders(
    *,
    data_dir: str | Path,
    batch_size: int = 128,
    num_workers: int = 0,
    download: bool = True,
    seed: int = 42,
) -> MNISTDataLoaders:
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    training_dataset = datasets.MNIST(
        root=str(root),
        train=True,
        transform=MNIST_TRANSFORM,
        download=download,
    )
    test_dataset = datasets.MNIST(
        root=str(root),
        train=False,
        transform=MNIST_TRANSFORM,
        download=download,
    )
    training_subset, validation_subset = split_training_dataset(
        training_dataset,
        validation_size=5_000,
        seed=seed,
    )

    training_generator = torch.Generator().manual_seed(seed)
    return MNISTDataLoaders(
        train=DataLoader(
            training_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=training_generator,
        ),
        validation=DataLoader(
            validation_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        test=DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    )

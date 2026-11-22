import pytest
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset

from neurovision_api.ml.data import MNIST_TRANSFORM, split_training_dataset


class IndexDataset(Dataset):
    def __len__(self) -> int:
        return 60_000

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.tensor(index), index % 10


def test_mnist_training_split_is_seeded_and_isolated() -> None:
    dataset = IndexDataset()
    training, validation = split_training_dataset(dataset, seed=42)
    training_again, validation_again = split_training_dataset(dataset, seed=42)

    assert len(training) == 55_000
    assert len(validation) == 5_000
    assert isinstance(training, Subset)
    assert isinstance(validation, Subset)
    assert set(training.indices).isdisjoint(validation.indices)
    assert training.indices == training_again.indices
    assert validation.indices == validation_again.indices


def test_training_split_rejects_invalid_validation_size() -> None:
    with pytest.raises(ValueError):
        split_training_dataset(IndexDataset(), validation_size=60_000)


def test_training_split_rejects_dataset_without_length() -> None:
    with pytest.raises(TypeError, match="defined length"):
        split_training_dataset(Dataset())


def test_mnist_transform_applies_documented_normalization() -> None:
    normalized = MNIST_TRANSFORM(Image.new("L", (28, 28), color=0))
    expected_black_pixel = (0.0 - 0.1307) / 0.3081

    assert normalized.shape == (1, 28, 28)
    assert normalized[0, 0, 0].item() == pytest.approx(expected_black_pixel)

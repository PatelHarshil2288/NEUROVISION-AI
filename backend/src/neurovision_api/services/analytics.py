from __future__ import annotations

import hashlib
import json
import math
import threading
from collections import Counter
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from torchvision import datasets

from neurovision_api.services.inference import InferenceService

HASH_PATTERN = r"^[0-9a-f]{64}$"


class Misclassification(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    sample_index: int = Field(ge=0)
    true_label: int = Field(ge=0, le=9)
    predicted_label: int = Field(ge=0, le=9)
    confidence: float = Field(ge=0, le=1)


class EvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: int = Field(ge=1, le=1)
    model_version: str
    checkpoint_sha256: str = Field(pattern=HASH_PATTERN)
    dataset: str
    dataset_split: str
    dataset_images_sha256: str = Field(pattern=HASH_PATTERN)
    dataset_labels_sha256: str = Field(pattern=HASH_PATTERN)
    generated_at: str
    loss: float = Field(ge=0)
    accuracy: float = Field(ge=0, le=1)
    precision_per_class: list[float] = Field(min_length=10, max_length=10)
    recall_per_class: list[float] = Field(min_length=10, max_length=10)
    f1_per_class: list[float] = Field(min_length=10, max_length=10)
    support_per_class: list[int] = Field(min_length=10, max_length=10)
    confusion_matrix: list[list[int]] = Field(min_length=10, max_length=10)
    misclassified_examples: list[Misclassification]
    num_samples: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        matrix = self.confusion_matrix
        if self.dataset != "MNIST" or self.dataset_split != "official-test":
            raise ValueError("The report must use the official MNIST test split.")
        if any(len(row) != 10 or any(value < 0 for value in row) for row in matrix):
            raise ValueError("The confusion matrix must contain non-negative 10 by 10 counts.")
        supports = [sum(row) for row in matrix]
        if supports != self.support_per_class or sum(supports) != self.num_samples:
            raise ValueError("Confusion matrix support does not match the sample count.")
        correct = sum(matrix[index][index] for index in range(10))
        if not math.isclose(self.accuracy, correct / self.num_samples, abs_tol=1e-8):
            raise ValueError("Accuracy does not match the confusion matrix.")
        for index in range(10):
            tp = matrix[index][index]
            predicted = sum(row[index] for row in matrix)
            precision = tp / predicted if predicted else 0
            recall = tp / supports[index] if supports[index] else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
            if not all(math.isclose(actual, expected, abs_tol=1e-8) for actual, expected in (
                (self.precision_per_class[index], precision),
                (self.recall_per_class[index], recall),
                (self.f1_per_class[index], f1),
            )):
                raise ValueError("Per-class metrics do not match the confusion matrix.")
        errors = self.misclassified_examples
        if len({item.sample_index for item in errors}) != len(errors):
            raise ValueError("Misclassification indices must be unique.")
        if any(item.sample_index >= self.num_samples or item.true_label == item.predicted_label for item in errors):
            raise ValueError("Misclassification entries are invalid.")
        counts = Counter((item.true_label, item.predicted_label) for item in errors)
        if any(counts[(row, column)] != matrix[row][column]
               for row in range(10) for column in range(10) if row != column):
            raise ValueError("Misclassifications do not match the confusion matrix.")
        return self


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_dataset_hashes(root: Path) -> tuple[str, str]:
    raw = root / "MNIST" / "raw"
    return (
        file_sha256(raw / "t10k-images-idx3-ubyte"),
        file_sha256(raw / "t10k-labels-idx1-ubyte"),
    )


class AnalyticsStore:
    def __init__(self, *, report_path: Path, data_dir: Path) -> None:
        self.report_path = report_path
        self.data_dir = data_dir
        self._lock = threading.Lock()
        self._dataset: datasets.MNIST | None = None
        self._dataset_hashes: tuple[str, str] | None = None

    def report(self, service: InferenceService) -> EvaluationReport:
        if not service.checkpoint_sha256:
            raise ValueError("The loaded model has no checkpoint identity.")
        if not self.report_path.is_file():
            raise FileNotFoundError("No evaluation report is available. Generate one for the loaded checkpoint.")
        if self.report_path.stat().st_size > 5_000_000:
            raise ValueError("The evaluation report is too large.")
        report = EvaluationReport.model_validate(json.loads(self.report_path.read_text(encoding="utf-8")))
        if report.checkpoint_sha256 != service.checkpoint_sha256 or report.model_version != service.model_version:
            raise ValueError("The evaluation report does not match the loaded checkpoint. Generate a new report.")
        return report

    def error_image(self, report: EvaluationReport, sample_index: int) -> list[list[float]]:
        example = next((item for item in report.misclassified_examples if item.sample_index == sample_index), None)
        if example is None:
            raise LookupError("This sample is not a recorded misclassification.")
        expected = (report.dataset_images_sha256, report.dataset_labels_sha256)
        with self._lock:
            if self._dataset is None or self._dataset_hashes != expected:
                if test_dataset_hashes(self.data_dir) != expected:
                    raise ValueError("The local test dataset does not match the evaluation report.")
                dataset = datasets.MNIST(root=str(self.data_dir), train=False, download=False)
                self._dataset = dataset
                self._dataset_hashes = expected
            dataset = self._dataset
            if len(dataset) != report.num_samples or int(dataset.targets[sample_index]) != example.true_label:
                raise ValueError("The local test dataset does not match this recorded example.")
            return (dataset.data[sample_index].float() / 255.0).tolist()

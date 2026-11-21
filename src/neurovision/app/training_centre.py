from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import torch
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from neurovision.ml.checkpoints import load_checkpoint, save_checkpoint
from neurovision.ml.data import mnist_dataloaders
from neurovision.ml.model import MLP, MLPConfig
from neurovision.ml.train import TrainProgress, train_epochs


class TrainingCentre(QWidget):
    model_updated = Signal(object)  # MLP
    progress_message = Signal(str)
    status_message = Signal(str)
    error_message = Signal(str)

    def __init__(self, *, data_dir: Path, parent=None) -> None:
        super().__init__(parent)
        self._data_dir = data_dir

        self._log = QTextEdit()
        self._log.setReadOnly(True)

        self._status = QLabel('Idle')
        self._status.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self._model_cfg = MLPConfig()
        torch.manual_seed(0)
        self._model = MLP(self._model_cfg)

        self._stop_flag = False
        self._worker: threading.Thread | None = None
        self.progress_message.connect(self._append)
        self.status_message.connect(self._set_status)
        self.error_message.connect(self._append)

        train_btn = QPushButton('Train on MNIST (CPU)')
        train_btn.clicked.connect(self._start_train)

        stop_btn = QPushButton('Stop Training')
        stop_btn.clicked.connect(self._stop)

        save_btn = QPushButton('Save Checkpoint')
        save_btn.clicked.connect(self._save_ckpt)

        load_btn = QPushButton('Load Checkpoint')
        load_btn.clicked.connect(self._load_ckpt)

        top = QHBoxLayout()
        top.addWidget(train_btn)
        top.addWidget(stop_btn)
        top.addWidget(save_btn)
        top.addWidget(load_btn)

        box = QGroupBox('Training Centre')
        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addWidget(self._status)
        layout.addWidget(self._log, 1)
        box.setLayout(layout)

        root = QVBoxLayout()
        root.addWidget(box)
        self.setLayout(root)

    @property
    def model(self) -> MLP:
        return self._model

    def _append(self, text: str) -> None:
        self._log.append(text)

    def _set_status(self, text: str) -> None:
        self._status.setText(text)

    def _start_train(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            self._append('Training already running.')
            return

        self._stop_flag = False
        self._log.clear()
        self._set_status('Preparing MNIST loaders...')

        def run() -> None:
            try:
                train_dl, test_dl = mnist_dataloaders(
                    data_dir=self._data_dir,
                    batch_size=128,
                    num_workers=0,
                    download=True,
                )

                device = torch.device('cpu')

                def on_progress(p: TrainProgress) -> None:
                    msg = f"epoch={p.epoch} step={p.step}/{p.steps_per_epoch} loss={p.train_loss:.4f}"
                    if p.test_accuracy is not None:
                        msg += f" test_acc={p.test_accuracy*100:.2f}%"
                    self.progress_message.emit(msg)
                    self.status_message.emit(msg)

                def should_stop() -> bool:
                    return self._stop_flag

                self.progress_message.emit('Starting training...')
                train_epochs(
                    model=self._model,
                    train_dl=train_dl,
                    test_dl=test_dl,
                    device=device,
                    epochs=3,
                    lr=1e-3,
                    on_progress=on_progress,
                    should_stop=should_stop,
                )

                if not self._stop_flag:
                    self.progress_message.emit('Training finished.')
                    self.status_message.emit('Training finished.')
                else:
                    self.progress_message.emit('Training stopped.')
                    self.status_message.emit('Training stopped.')

                self.model_updated.emit(self._model)
            except Exception as e:
                self.error_message.emit(f'ERROR: {e!r}')
                self.status_message.emit('Error')

        self._worker = threading.Thread(target=run, daemon=True)
        self._worker.start()

    def _stop(self) -> None:
        self._stop_flag = True

    def _save_ckpt(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, 'Save Checkpoint', str(self._data_dir / 'checkpoints'), 'PyTorch (*.pt)')
        if not path:
            return
        save_checkpoint(path, model=self._model, config=self._model_cfg)
        self._append(f'Saved checkpoint: {path}')

    def _load_ckpt(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, 'Load Checkpoint', str(self._data_dir / 'checkpoints'), 'PyTorch (*.pt)')
        if not path:
            return
        model, cfg = load_checkpoint(path)
        self._model = model
        self._model_cfg = cfg
        self._append(f'Loaded checkpoint: {path}')
        self.model_updated.emit(self._model)

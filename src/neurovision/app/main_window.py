from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from neurovision.app.digit_studio import DigitStudio, DigitStudioResult
from neurovision.app.training_centre import TrainingCentre
from neurovision.ml.checkpoints import load_checkpoint
from neurovision.ml.inference import forward_with_telemetry_model
from neurovision.ml.model import MLP, MLPConfig
from neurovision.renderer.config import RenderConfig
from neurovision.renderer.gl_widget import NeuralGLWidget
from neurovision.renderer.state import ModelState, build_state


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle('NEUROVISION AI')
        self.resize(1200, 760)

        self._log = QTextEdit()
        self._log.setReadOnly(True)

        self._status = QLabel('Ready')
        self._status.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self._render_cfg = RenderConfig()
        self._state: ModelState | None = None
        data_dir = Path.cwd() / 'data'

        self._model_cfg = MLPConfig(input_dim=784, hidden1=128, hidden2=64, output_dim=10)
        self._model = self._new_model(self._model_cfg)
        default_checkpoint = data_dir / 'checkpoints' / 'mnist.pt'
        if default_checkpoint.is_file():
            try:
                self._model, self._model_cfg = load_checkpoint(default_checkpoint)
                self._status = QLabel(f'Model loaded: {default_checkpoint.name}')
                self._status.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            except Exception as exc:
                self._status = QLabel(f'Default checkpoint could not be loaded: {exc}')
                self._status.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self._renderer = NeuralGLWidget()

        tabs = QTabWidget()
        tabs.addTab(self._make_command_centre(), 'Command Centre')
        tabs.addTab(self._make_digit_studio(), 'Digit Studio')

        self._training = TrainingCentre(data_dir=data_dir)
        self._training.model_updated.connect(self._on_model_updated)
        tabs.addTab(self._training, 'Training Centre')

        right = QVBoxLayout()
        right.addWidget(QLabel('3D Neural Observatory'))
        right.addWidget(self._renderer, 1)

        root = QHBoxLayout()
        left_w = QWidget()
        left_w.setLayout(QVBoxLayout())
        left_w.layout().addWidget(tabs)
        left_w.layout().addWidget(self._status)
        left_w.layout().addWidget(QLabel('Telemetry'))
        left_w.layout().addWidget(self._log, 1)

        right_w = QWidget()
        right_w.setLayout(right)

        root.addWidget(left_w, 2)
        root.addWidget(right_w, 3)

        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

    def _new_model(self, cfg: MLPConfig) -> MLP:
        # Keep deterministic init unless a checkpoint or training updates it.
        import torch

        torch.manual_seed(0)
        model = MLP(cfg)
        model.eval()
        return model

    def _on_model_updated(self, model: MLP) -> None:
        self._model = model
        self._status.setText('Model updated (training/checkpoint).')

    def _make_command_centre(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout()

        run_btn = QPushButton('Run Forward Pass (Dummy 28x28)')
        run_btn.clicked.connect(self._run_forward_dummy)

        play_btn = QPushButton('Play Propagation (Telemetry Driven)')
        play_btn.clicked.connect(self._play)

        stop_btn = QPushButton('Stop')
        stop_btn.clicked.connect(self._stop)

        layout.addWidget(run_btn)
        layout.addWidget(play_btn)
        layout.addWidget(stop_btn)
        layout.addStretch(1)
        w.setLayout(layout)
        return w

    def _make_digit_studio(self) -> QWidget:
        studio = DigitStudio()
        studio.image_ready.connect(self._run_forward_from_studio)
        return studio

    def _append(self, text: str) -> None:
        self._log.append(text)

    def _run_forward_dummy(self) -> None:
        image = np.zeros((28, 28), dtype=np.float32)
        self._run_forward(image)

    def _run_forward_from_studio(self, payload: DigitStudioResult) -> None:
        self._run_forward(payload.mnist_image.astype(np.float32))

    def _run_forward(self, image_28x28: np.ndarray) -> None:
        result = forward_with_telemetry_model(self._model, image_28x28)
        pred = int(result.probabilities.argmax())

        self._status.setText(f'Prediction: {pred} | probs sum={result.probabilities.sum():.4f}')
        self._log.clear()

        self._append('=== Forward Pass ===')
        self._append(f'prediction={pred}')
        self._append(f'probabilities={np.array2string(result.probabilities, precision=4)}')
        self._append('')

        self._append('=== Telemetry Events ===')
        for event in result.events:
            self._append(str(asdict(event)))

        self._state = build_state(
            image_flat=result.image_flat,
            h1=result.h1_activation,
            h2=result.h2_activation,
            probs=result.probabilities,
            w1=result.w1,
            w2=result.w2,
            w3=result.w3,
            max_edges_per_pair=self._render_cfg.max_edges_per_pair,
        )
        self._renderer.set_state(self._state)

    def _play(self) -> None:
        if self._state is None:
            self._status.setText('Run a forward pass first.')
            return
        self._renderer.play_propagation()

    def _stop(self) -> None:
        self._renderer.stop()

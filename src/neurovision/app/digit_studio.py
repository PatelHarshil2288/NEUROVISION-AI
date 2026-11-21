from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from neurovision.digit.preprocess import canvas_to_mnist


@dataclass(frozen=True)
class DigitStudioResult:
    mnist_image: np.ndarray  # (28,28) float


class DigitCanvas(QWidget):
    changed = Signal()

    def __init__(self, size_px: int = 280, parent=None) -> None:
        super().__init__(parent)
        self._size = size_px
        self.setFixedSize(size_px, size_px)

        self._img = QImage(size_px, size_px, QImage.Format_ARGB32)
        self._img.fill(QColor(255, 255, 255))

        self._pen = QPen(QColor(0, 0, 0))
        self._pen.setWidth(22)
        self._pen.setCapStyle(Qt.RoundCap)
        self._pen.setJoinStyle(Qt.RoundJoin)

        self._last: QPoint | None = None

    def clear(self) -> None:
        self._img.fill(QColor(255, 255, 255))
        self.changed.emit()
        self.update()

    def to_ink_numpy(self) -> np.ndarray:
        # Convert to grayscale and map: black ink -> 1.0, white background -> 0.0
        g = self._img.convertToFormat(QImage.Format_Grayscale8)
        w = g.width()
        h = g.height()
        ptr = g.bits()
        arr = np.frombuffer(ptr.tobytes(), dtype=np.uint8).reshape((h, w))
        ink = 1.0 - (arr.astype(np.float32) / 255.0)
        return np.clip(ink, 0.0, 1.0)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.drawImage(0, 0, self._img)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.LeftButton:
            return
        pos = event.position().toPoint()

        painter = QPainter(self._img)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0))
        radius = self._pen.width() // 2
        painter.drawEllipse(pos, radius, radius)
        painter.end()

        self._last = pos
        self.changed.emit()
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._last is None:
            return
        if not (event.buttons() & Qt.LeftButton):
            return

        pos = event.position().toPoint()
        painter = QPainter(self._img)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(self._pen)
        painter.drawLine(self._last, pos)
        painter.end()

        self._last = pos
        self.changed.emit()
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._last = None


class DigitStudio(QWidget):
    # Emits a ready-to-run MNIST-like image (28x28 float) from real user drawing.
    image_ready = Signal(object)  # DigitStudioResult

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._canvas = DigitCanvas()
        self._preview = QLabel()
        self._preview.setFixedSize(140, 140)
        self._preview.setAlignment(Qt.AlignCenter)

        clear_btn = QPushButton('Clear')
        clear_btn.clicked.connect(self._canvas.clear)

        emit_btn = QPushButton('Use Drawing For Inference')
        emit_btn.clicked.connect(self._emit_image)

        self._canvas.changed.connect(self._update_preview)
        self._update_preview()

        preview_box = QGroupBox('Preprocessed (28x28)')
        pv = QVBoxLayout()
        pv.addWidget(self._preview)
        preview_box.setLayout(pv)

        controls = QHBoxLayout()
        controls.addWidget(clear_btn)
        controls.addWidget(emit_btn)

        left = QVBoxLayout()
        left.addWidget(QLabel('Draw a digit (mouse)'))
        left.addWidget(self._canvas)
        left.addLayout(controls)

        right = QVBoxLayout()
        right.addWidget(preview_box)
        right.addStretch(1)

        root = QHBoxLayout()
        root.addLayout(left)
        root.addLayout(right)
        self.setLayout(root)

    def _emit_image(self) -> None:
        ink = self._canvas.to_ink_numpy()
        mnist = canvas_to_mnist(ink)
        self.image_ready.emit(DigitStudioResult(mnist_image=mnist))

    def _update_preview(self) -> None:
        ink = self._canvas.to_ink_numpy()
        mnist = canvas_to_mnist(ink)

        # Show as grayscale where ink is bright.
        img_u8 = (mnist * 255.0).clip(0, 255).astype(np.uint8)
        qimg = QImage(img_u8.data, 28, 28, 28, QImage.Format_Grayscale8)
        pix = QPixmap.fromImage(qimg.copy()).scaled(
            self._preview.width(),
            self._preview.height(),
            Qt.KeepAspectRatio,
            Qt.FastTransformation,
        )
        self._preview.setPixmap(pix)

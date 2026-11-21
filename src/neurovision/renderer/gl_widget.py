from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QPoint, QTimer, Qt
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from neurovision.renderer.config import CameraState, RenderConfig
from neurovision.renderer.math3d import look_at, perspective, rotate_x, rotate_y
from neurovision.renderer.state import ModelState


try:
    from OpenGL.GL import (  # type: ignore
        GL_COLOR_BUFFER_BIT,
        GL_DEPTH_BUFFER_BIT,
        GL_DEPTH_TEST,
        GL_LINES,
        GL_MODELVIEW,
        GL_POINTS,
        GL_PROJECTION,
        glBegin,
        glClear,
        glClearColor,
        glColor3f,
        glEnable,
        glEnd,
        glLineWidth,
        glLoadMatrixf,
        glMatrixMode,
        glPointSize,
        glVertex3f,
        glViewport,
    )
except Exception:  # pragma: no cover
    glBegin = None


def _to_gl(m: np.ndarray) -> np.ndarray:
    # OpenGL expects column-major.
    return np.asarray(m.T, dtype=np.float32)


def _color_from_activation(a: float) -> tuple[float, float, float]:
    # a in [-1, 1] or [0, 1] for output
    if a < 0.0:
        return (0.55, 0.25, 0.85)
    t = min(1.0, a)
    return (0.15 + 0.10 * t, 0.65 + 0.35 * t, 0.75 + 0.25 * t)


class NeuralGLWidget(QOpenGLWidget):
    def __init__(self, parent=None) -> None:
        fmt = QSurfaceFormat()
        fmt.setRenderableType(QSurfaceFormat.OpenGL)
        fmt.setVersion(2, 1)
        fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
        fmt.setDepthBufferSize(24)
        QSurfaceFormat.setDefaultFormat(fmt)

        super().__init__(parent)

        self.setFocusPolicy(Qt.StrongFocus)

        self._camera = CameraState()
        self._cfg = RenderConfig()
        self._state: Optional[ModelState] = None

        self._last_pos: Optional[QPoint] = None

        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(30)
        self._anim_timer.timeout.connect(self._tick)
        self._anim_t = 0.0
        self._anim_on = False

    def set_state(self, state: Optional[ModelState]) -> None:
        self._state = state
        self.update()

    def play_propagation(self) -> None:
        self._anim_t = 0.0
        self._anim_on = True
        self._anim_timer.start()

    def stop(self) -> None:
        self._anim_on = False
        self._anim_timer.stop()
        self.update()

    def _tick(self) -> None:
        self._anim_t += 0.03
        if self._anim_t >= 4.0:
            self.stop()
            return
        self.update()

    # Interaction
    def mousePressEvent(self, event) -> None:
        self._last_pos = event.position().toPoint()

    def mouseMoveEvent(self, event) -> None:
        if self._last_pos is None:
            return
        p = event.position().toPoint()
        dx = p.x() - self._last_pos.x()
        dy = p.y() - self._last_pos.y()
        self._last_pos = p

        self._camera = CameraState(
            yaw_deg=self._camera.yaw_deg + dx * 0.4,
            pitch_deg=max(-89.0, min(89.0, self._camera.pitch_deg + dy * 0.4)),
            distance=self._camera.distance,
        )
        self.update()

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y() / 120.0
        dist = max(4.0, min(40.0, self._camera.distance * (0.90 ** delta)))
        self._camera = CameraState(
            yaw_deg=self._camera.yaw_deg,
            pitch_deg=self._camera.pitch_deg,
            distance=dist,
        )
        self.update()

    # GL
    def initializeGL(self) -> None:
        if glBegin is None:
            return
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.02, 0.03, 0.05, 1.0)

    def resizeGL(self, w: int, h: int) -> None:
        if glBegin is None:
            return
        glViewport(0, 0, max(1, w), max(1, h))

    def paintGL(self) -> None:
        if glBegin is None:
            return

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        w = max(1, self.width())
        h = max(1, self.height())
        aspect = float(w) / float(h)

        proj = perspective(55.0, aspect, 0.1, 200.0).m

        # Orbit camera around origin
        eye = np.array([0.0, 0.0, self._camera.distance], dtype=np.float32)
        orbit = (rotate_y(self._camera.yaw_deg) @ rotate_x(self._camera.pitch_deg)).m
        eye_h = orbit @ np.array([eye[0], eye[1], eye[2], 1.0], dtype=np.float32)
        eye_v = eye_h[:3]
        view = look_at(
            eye_v,
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        ).m

        glMatrixMode(GL_PROJECTION)
        glLoadMatrixf(_to_gl(proj))
        glMatrixMode(GL_MODELVIEW)
        glLoadMatrixf(_to_gl(view))

        if self._state is None:
            return

        stage = min(3, int(self._anim_t)) if self._anim_on else 3

        self._draw_edges(stage)
        self._draw_nodes(stage)

    def _draw_nodes(self, stage: int) -> None:
        assert self._state is not None
        st = self._state

        # Input
        glPointSize(self._cfg.point_size_input)
        glBegin(GL_POINTS)
        if stage >= 0:
            for pos, a in zip(st.layout.input_pos, st.input_activation, strict=False):
                r, g, b = _color_from_activation(float(a))
                glColor3f(r, g, b)
                glVertex3f(float(pos[0]), float(pos[1]), float(pos[2]))
        glEnd()

        # Hidden 1
        glPointSize(self._cfg.point_size_hidden)
        glBegin(GL_POINTS)
        if stage >= 1:
            for pos, a in zip(st.layout.h1_pos, st.h1_activation, strict=False):
                r, g, b = _color_from_activation(float(a))
                glColor3f(r, g, b)
                glVertex3f(float(pos[0]), float(pos[1]), float(pos[2]))
        else:
            for pos in st.layout.h1_pos:
                glColor3f(0.12, 0.18, 0.25)
                glVertex3f(float(pos[0]), float(pos[1]), float(pos[2]))
        glEnd()

        # Hidden 2
        glPointSize(self._cfg.point_size_hidden)
        glBegin(GL_POINTS)
        if stage >= 2:
            for pos, a in zip(st.layout.h2_pos, st.h2_activation, strict=False):
                r, g, b = _color_from_activation(float(a))
                glColor3f(r, g, b)
                glVertex3f(float(pos[0]), float(pos[1]), float(pos[2]))
        else:
            for pos in st.layout.h2_pos:
                glColor3f(0.10, 0.16, 0.22)
                glVertex3f(float(pos[0]), float(pos[1]), float(pos[2]))
        glEnd()

        # Output
        glPointSize(self._cfg.point_size_output)
        glBegin(GL_POINTS)
        if stage >= 3:
            for pos, p in zip(st.layout.out_pos, st.out_activation, strict=False):
                r, g, b = _color_from_activation(float(p))
                glColor3f(r, g, b)
                glVertex3f(float(pos[0]), float(pos[1]), float(pos[2]))
        else:
            for pos in st.layout.out_pos:
                glColor3f(0.10, 0.14, 0.18)
                glVertex3f(float(pos[0]), float(pos[1]), float(pos[2]))
        glEnd()

    def _draw_edges(self, stage: int) -> None:
        assert self._state is not None
        st = self._state

        glLineWidth(1.0)
        glBegin(GL_LINES)

        if stage >= 1:
            glColor3f(0.10, 0.55, 0.65)
            self._draw_edge_set(st.layout.input_pos, st.layout.h1_pos, st.edges_in_h1)

        if stage >= 2:
            glColor3f(0.15, 0.45, 0.75)
            self._draw_edge_set(st.layout.h1_pos, st.layout.h2_pos, st.edges_h1_h2)

        if stage >= 3:
            glColor3f(0.35, 0.25, 0.85)
            self._draw_edge_set(st.layout.h2_pos, st.layout.out_pos, st.edges_h2_out)

        glEnd()

    def _draw_edge_set(self, src_pos: np.ndarray, dst_pos: np.ndarray, edges: np.ndarray) -> None:
        if edges.size == 0:
            return

        for src_i, dst_i in edges:
            sp = src_pos[int(src_i)]
            dp = dst_pos[int(dst_i)]
            glVertex3f(float(sp[0]), float(sp[1]), float(sp[2]))
            glVertex3f(float(dp[0]), float(dp[1]), float(dp[2]))

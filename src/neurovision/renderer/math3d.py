from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Mat4:
    m: np.ndarray  # shape (4, 4)

    def __matmul__(self, other: 'Mat4') -> 'Mat4':
        return Mat4(self.m @ other.m)


def identity() -> Mat4:
    return Mat4(np.eye(4, dtype=np.float32))


def perspective(fov_y_deg: float, aspect: float, z_near: float, z_far: float) -> Mat4:
    f = 1.0 / math.tan(math.radians(fov_y_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / max(aspect, 1e-6)
    m[1, 1] = f
    m[2, 2] = (z_far + z_near) / (z_near - z_far)
    m[2, 3] = (2.0 * z_far * z_near) / (z_near - z_far)
    m[3, 2] = -1.0
    return Mat4(m)


def translate(x: float, y: float, z: float) -> Mat4:
    m = np.eye(4, dtype=np.float32)
    m[0, 3] = x
    m[1, 3] = y
    m[2, 3] = z
    return Mat4(m)


def rotate_y(deg: float) -> Mat4:
    a = math.radians(deg)
    c = math.cos(a)
    s = math.sin(a)
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = c
    m[0, 2] = s
    m[2, 0] = -s
    m[2, 2] = c
    return Mat4(m)


def rotate_x(deg: float) -> Mat4:
    a = math.radians(deg)
    c = math.cos(a)
    s = math.sin(a)
    m = np.eye(4, dtype=np.float32)
    m[1, 1] = c
    m[1, 2] = -s
    m[2, 1] = s
    m[2, 2] = c
    return Mat4(m)


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> Mat4:
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)

    f = target - eye
    f = f / max(float(np.linalg.norm(f)), 1e-6)
    u = up / max(float(np.linalg.norm(up)), 1e-6)
    s = np.cross(f, u)
    s = s / max(float(np.linalg.norm(s)), 1e-6)
    u2 = np.cross(s, f)

    m = np.eye(4, dtype=np.float32)
    m[0, 0:3] = s
    m[1, 0:3] = u2
    m[2, 0:3] = -f
    m[0, 3] = -float(np.dot(s, eye))
    m[1, 3] = -float(np.dot(u2, eye))
    m[2, 3] = float(np.dot(f, eye))
    return Mat4(m)

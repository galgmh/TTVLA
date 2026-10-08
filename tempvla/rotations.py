"""Pure-numpy quaternion helpers (wxyz, Hamilton product).

Self-contained so action math does not depend on engine-specific broadcasting
rules; validated against genesis.utils.geom at Gate 1.
"""

from __future__ import annotations

import numpy as np


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return q / np.linalg.norm(q)


def quat_mul(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Hamilton product u ⊗ v (rotation by v first, then u)."""
    w1, x1, y1, z1 = np.asarray(u, dtype=np.float64).reshape(4)
    w2, x2, y2, z2 = np.asarray(v, dtype=np.float64).reshape(4)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_rotate(q: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate a 3-vector by quaternion q."""
    v = np.asarray(vec, dtype=np.float64).reshape(3)
    return quat_mul(quat_mul(q, np.array([0.0, *v])), quat_conj(q))[1:]


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """wxyz quaternion -> (3,3) rotation matrix, same convention as quat_rotate.

    Accepts a leading batch: ``(..., 4) -> (..., 3, 3)``. Exists so a whole
    (N,3) block can be rotated with one matmul instead of N pure-python Hamilton
    products — the 512-marker gel field made that loop the dominant per-step
    cost — and so readers have the exact inverse to hand: ``v_link = Rᵀ @ v_world``.
    """
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    rows = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    )
    return np.stack([np.stack(r, axis=-1) for r in rows], axis=-2)


def rotvec_to_quat(rotvec: np.ndarray) -> np.ndarray:
    """Axis-angle (angle * unit axis) -> wxyz quaternion."""
    rv = np.asarray(rotvec, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(rv))
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = rv / angle
    return np.concatenate([[np.cos(0.5 * angle)], np.sin(0.5 * angle) * axis])


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """wxyz quaternion -> axis-angle vector (shortest arc, w >= 0)."""
    q = quat_normalize(q)
    if q[0] < 0.0:
        q = -q
    vec = q[1:]
    s = float(np.linalg.norm(vec))
    if s < 1e-12:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(s, float(q[0]))
    return vec / s * angle


def quat_error_rotvec(q_target: np.ndarray, q_current: np.ndarray) -> np.ndarray:
    """World-frame rotation vector taking q_current to q_target."""
    return quat_to_rotvec(quat_mul(q_target, quat_conj(q_current)))


def quat_apply_delta(q_current: np.ndarray, delta_rotvec: np.ndarray) -> np.ndarray:
    """World-frame delta rotation composed onto the current orientation."""
    return quat_normalize(quat_mul(rotvec_to_quat(delta_rotvec), q_current))


def yaw_to_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)])


def yaw_of_quat(q: np.ndarray) -> float:
    """Yaw (rotation about world z) of a wxyz quaternion, in radians.

    Exact for the yaw-only / yaw-with-flip poses this project builds; for an
    arbitrary orientation it is the heading of the rotated x-axis, which is the
    quantity the box face-snap needs.
    """
    q = quat_normalize(np.asarray(q, dtype=np.float64).reshape(-1)[:4])
    w, x, y, z = (float(v) for v in q)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

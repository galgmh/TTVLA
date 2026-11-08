"""纯 numpy 的四元数工具（wxyz 顺序，Hamilton 乘法）。

动作/观测链路不依赖引擎特有的广播规则，因此这里自带一份实现；
已用 genesis.utils.geom 逐项核对过。
"""

from __future__ import annotations

import numpy as np


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """单位化，输入按长度 4 解释。"""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return q / np.linalg.norm(q)


def quat_mul(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Hamilton 乘积 u ⊗ v（先施加 v，再施加 u）。"""
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
    """共轭（逆旋转）。"""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_rotate(q: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """用四元数旋转三维向量。"""
    v = np.asarray(vec, dtype=np.float64).reshape(3)
    return quat_mul(quat_mul(q, np.array([0.0, *v])), quat_conj(q))[1:]


def rotvec_to_quat(rotvec: np.ndarray) -> np.ndarray:
    """轴角向量（角度 × 单位轴）-> wxyz 四元数。"""
    rv = np.asarray(rotvec, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(rv))
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = rv / angle
    return np.concatenate([[np.cos(0.5 * angle)], np.sin(0.5 * angle) * axis])


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """wxyz 四元数 -> 轴角向量（取最短弧，w >= 0）。"""
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
    """把 q_current 转到 q_target 的世界系旋转向量。"""
    return quat_to_rotvec(quat_mul(q_target, quat_conj(q_current)))


def quat_apply_delta(q_current: np.ndarray, delta_rotvec: np.ndarray) -> np.ndarray:
    """在世界系把增量旋转叠加到当前朝向上。"""
    return quat_normalize(quat_mul(rotvec_to_quat(delta_rotvec), q_current))


def yaw_to_quat(yaw: float) -> np.ndarray:
    """绕世界 z 轴的偏航角 -> wxyz 四元数。"""
    return np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)])


__all__ = [
    "quat_apply_delta",
    "quat_conj",
    "quat_error_rotvec",
    "quat_mul",
    "quat_normalize",
    "quat_rotate",
    "quat_to_rotvec",
    "rotvec_to_quat",
    "yaw_to_quat",
]

"""热觉特征与指尖几何接口。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

# 温度斜率的平滑时间常数（秒）：原始 50 Hz 序列上的一阶低通。
RATE_SMOOTH_S = 0.1


@dataclass(frozen=True)
class FingertipGeometry:
    """Fixed-FK outputs for both pads, both shaped [B,2,3]."""

    positions_base: Any
    normals_base: Any


class FingertipGeometryProvider(Protocol):
    """Map raw robot state ``[B,8]`` to left/right pad geometry."""

    def __call__(self, state: Any) -> FingertipGeometry: ...


def _rate(values: np.ndarray, dt: float) -> np.ndarray:
    """一阶低通 + 中心差分得到温度斜率 (°C/s)。"""
    alpha = min(1.0, dt / max(RATE_SMOOTH_S, dt))
    smooth = np.empty_like(values)
    smooth[0] = values[0]
    for i in range(1, len(values)):
        smooth[i] = smooth[i - 1] + alpha * (values[i] - smooth[i - 1])
    rate = np.zeros_like(values)
    if len(values) > 2:
        rate[1:-1] = (smooth[2:] - smooth[:-2]) / (2.0 * dt)
        rate[0] = (smooth[1] - smooth[0]) / dt
        rate[-1] = (smooth[-1] - smooth[-2]) / dt
    return rate


def frame_features(arrays: dict[str, np.ndarray], *, dt: float) -> np.ndarray:
    """逐帧 12 维特征矩阵 (L, 12)，通道顺序见 ``ENCODER_INPUT_CHANNELS``。

    这里只派生真实机器人可从温度、时间戳和双指三轴力得到的量。
    接触面积、夹爪宽度和接触事件时长不进入底层 Thermal Encoder。
    """
    temp = np.asarray(arrays["tactile/fingertip_temp"], dtype=np.float64)
    force = np.asarray(arrays["tactile/contact_force_link"], dtype=np.float64)
    if force.ndim != 3 or force.shape[1:] != (2, 3):
        raise ValueError(f"tactile/contact_force_link 应为 [L,2,3]，实为 {force.shape}")

    dtemp = temp - temp[:1]
    rate = np.stack([_rate(temp[:, 0], dt), _rate(temp[:, 1], dt)], axis=1)

    force_channels = force.reshape(len(force), 6)  # left xyz, right xyz
    return np.concatenate([temp, dtemp, rate, force_channels], axis=1).astype(np.float32)


__all__ = ["FingertipGeometry", "FingertipGeometryProvider", "frame_features"]

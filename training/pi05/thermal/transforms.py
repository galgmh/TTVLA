"""数据批次 → Observation 热觉字段的映射（规格第 10 节，Task G）。

openpi 的 data transform 在 numpy 数据字典上运行。本模块提供：

* ``ThermalObservationTransform``：把数据集里的热觉数组统一映射/转换成
  ``Observation`` 需要的4个热觉字段（源键名可配置，默认同名）；
* ``thermal_supervision_from_batch``：从批次里的 ``thermal_supervision``
  子字典构造 ``ThermalSupervision``（监督不进入 Observation）；
* ``split_thermal_batch``：把监督子字典从批次中拆出来，供训练循环使用。
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
from typing import Any

import numpy as np
import torch

from models.pi05.thermal.outputs import ThermalSupervision
from training.pi05.thermal.labels import build_thermal_supervision

#: 热觉 Observation 字段（与规格第 4 节一致）。
THERMAL_OBSERVATION_KEYS = (
    "thermal_window",
    "thermal_window_time_mask",
    "sensor_xyz_base",
    "sensor_normal_base",
)

_BOOL_KEYS = frozenset(
    {
        "thermal_window_time_mask",
    }
)
FLOAT_KEYS = frozenset(THERMAL_OBSERVATION_KEYS) - _BOOL_KEYS

#: 默认源键名与 Observation 字段同名。
DEFAULT_THERMAL_KEY_MAP = {key: key for key in THERMAL_OBSERVATION_KEYS}


@dataclasses.dataclass(frozen=True)
class ThermalObservationTransform:
    """把源数据键映射到 Observation 热觉字段，并规范 dtype。

    ``key_map`` 的键是数据集里的原始键，值是 Observation 字段名。默认同名；
    若数据集用 ``thermal/history`` 之类的键，只需改写映射。``require_all=True``
    时缺任一字段立即报错——热觉开启时7个字段都必需。
    """

    key_map: Mapping[str, str] = dataclasses.field(
        default_factory=lambda: dict(DEFAULT_THERMAL_KEY_MAP)
    )
    require_all: bool = True

    def __call__(self, data: Mapping[str, Any]) -> dict[str, Any]:
        missing = [source for source in self.key_map if source not in data]
        if missing and self.require_all:
            raise ValueError(f"数据缺少热觉字段: {missing}")
        output = dict(data)
        for source, target in self.key_map.items():
            if source not in data:
                continue
            value = np.asarray(data[source])
            output[target] = value.astype(bool) if target in _BOOL_KEYS else value.astype(np.float32)
        return output


@dataclasses.dataclass(frozen=True)
class EmptyThermalObservation:
    """给没有热觉传感器的数据集补齐空接触历史。

    用于在 LIBERO 等纯视觉数据上微调 Thermal-π0.5：热觉分支收到合法的空历史
    （K个空槽），不会因为缺字段而报错。默认只补缺失键，
    ``overwrite=True`` 时覆盖已有字段。
    """

    window_size: int
    signal_dim: int
    overwrite: bool = False

    def _empty_fields(self) -> dict[str, np.ndarray]:
        length, channels = self.window_size, self.signal_dim
        return {
            "thermal_window": np.zeros((length, channels), dtype=np.float32),
            "thermal_window_time_mask": np.zeros((length,), dtype=bool),
            "sensor_xyz_base": np.zeros((3,), dtype=np.float32),
            "sensor_normal_base": np.zeros((3,), dtype=np.float32),
        }

    def __call__(self, data: Mapping[str, Any]) -> dict[str, Any]:
        output = dict(data)
        for key, value in self._empty_fields().items():
            if self.overwrite or key not in output:
                output[key] = value
        return output


def thermal_supervision_from_batch(
    batch: Mapping[str, Any],
    *,
    device: torch.device | str | None = None,
) -> ThermalSupervision | None:
    """从批次里取出 ``thermal_supervision`` 并转成模型监督容器。"""
    arrays = batch.get("thermal_supervision")
    if arrays is None:
        return None
    return build_thermal_supervision(arrays, device=device)


def split_thermal_batch(batch: Mapping[str, Any]) -> tuple[dict[str, Any], Mapping[str, Any] | None]:
    """返回 ``(去掉监督的批次, 监督数组或 None)``（不修改输入）。"""
    data = dict(batch)
    supervision = data.pop("thermal_supervision", None)
    return data, supervision


__all__ = [
    "DEFAULT_THERMAL_KEY_MAP",
    "FLOAT_KEYS",
    "THERMAL_OBSERVATION_KEYS",
    "EmptyThermalObservation",
    "ThermalObservationTransform",
    "split_thermal_batch",
    "thermal_supervision_from_batch",
]

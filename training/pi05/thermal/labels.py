"""仿真特权标签 → ``ThermalSupervision``（规格第 5 节，Task G）。

标签只参与损失，绝不进入 ``Observation``。本模块提供两个层次的工具：

1. 几何/温度标签：用区域中心最近邻为训练/评测查询坐标分配温度；
2. ``build_thermal_supervision``：把 numpy/torch 数组整理成模型 loss 需要的
   ``ThermalSupervision``，统一 dtype 并做范围校验。
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

from models.pi05.thermal.outputs import ThermalSupervision
from models.pi05.thermal.strategy_aux import SKILL_ABORT
from models.pi05.thermal.strategy_aux import SKILL_GRASP
from models.pi05.thermal.strategy_aux import SKILL_PROBE
from models.pi05.thermal.strategy_aux import SKILL_RETRACT

#: 监督字段名（与 ``ThermalSupervision`` 字段一致）。
SUPERVISION_FIELDS = (
    "contact_temp_c",
    "contact_valid",
    "field_query_xyz_base",
    "field_query_temp_c",
    "field_query_mask",
    "strategy_type",
)

#: 技能编号固定为 0=PROBE, 1=GRASP, 2=RETRACT, 3=ABORT。


def assign_region_temperatures(
    xyz_base: np.ndarray,
    region_centers_base: np.ndarray,
    region_temps_c: np.ndarray,
    *,
    radius_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """最近区域中心分配温度；超出半径的点标为无效。

    Args:
        xyz_base: [N, 3] 训练/评测查询点（机器人基座坐标系）。
        region_centers_base: [R, 3] 各热区中心（机器人基座坐标系）。
        region_temps_c: [R] 各热区当前温度。
        radius_m: 点属于某热区的最大中心距。

    Returns:
        ``(temps_c [N], mask [N])``；无效点的温度置 0，由 mask 排除出损失。
    """
    points = np.asarray(xyz_base, dtype=np.float32)
    centers = np.asarray(region_centers_base, dtype=np.float32)
    temps = np.asarray(region_temps_c, dtype=np.float32).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"xyz_base 必须为 [N, 3]，得到 {points.shape}")
    if centers.ndim != 2 or centers.shape[1] != 3 or centers.shape[0] == 0:
        raise ValueError(f"region_centers_base 必须为 [R, 3] 且 R>0，得到 {centers.shape}")
    if temps.shape[0] != centers.shape[0]:
        raise ValueError(f"region_temps_c 长度 {temps.shape[0]} 与区域数 {centers.shape[0]} 不一致")
    if radius_m <= 0:
        raise ValueError("radius_m 必须 > 0")

    distance = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=-1)  # [N, R]
    nearest = np.argmin(distance, axis=1)
    mask = np.min(distance, axis=1) <= float(radius_m)
    values = temps[nearest].astype(np.float32)
    values[~mask] = 0.0
    return values, mask.astype(bool)


def _to_tensor(value, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(value), dtype=dtype, device=device)


def build_thermal_supervision(
    arrays: Mapping[str, object] | None = None,
    *,
    device: torch.device | str | None = None,
    **kwargs,
) -> ThermalSupervision:
    """把数组/mapping 组装成 ``ThermalSupervision``。

    支持 ``build_thermal_supervision({...})`` 或
    ``build_thermal_supervision(contact_temp_c=..., ...)`` 两种写法；
    允许带任意前导 batch 维（逐样本或已批处理均可）。
    """
    if arrays is not None and kwargs:
        raise ValueError("arrays 与关键字参数只能二选一")
    values = dict(arrays) if arrays is not None else dict(kwargs)
    missing = [name for name in SUPERVISION_FIELDS if name not in values]
    if missing:
        raise ValueError(f"缺少监督字段: {missing}")

    if device is None:
        for value in values.values():
            if isinstance(value, torch.Tensor):
                device = value.device
                break
    device = torch.device("cpu") if device is None else torch.device(device)

    supervision = ThermalSupervision(
        contact_temp_c=_to_tensor(values["contact_temp_c"], dtype=torch.float32, device=device),
        contact_valid=_to_tensor(values["contact_valid"], dtype=torch.float32, device=device),
        field_query_xyz_base=_to_tensor(values["field_query_xyz_base"], dtype=torch.float32, device=device),
        field_query_temp_c=_to_tensor(values["field_query_temp_c"], dtype=torch.float32, device=device),
        field_query_mask=_to_tensor(values["field_query_mask"], dtype=torch.bool, device=device),
        strategy_type=_to_tensor(values["strategy_type"], dtype=torch.long, device=device),
    )
    _validate_supervision(supervision)
    return supervision


def _validate_supervision(supervision: ThermalSupervision) -> None:
    """形状与技能编号校验（逐样本或带batch维均可）。"""
    contact_shape = supervision.contact_temp_c.shape
    if supervision.contact_valid.shape != contact_shape:
        raise ValueError(
            f"contact_valid 形状 {tuple(supervision.contact_valid.shape)} 与 contact_temp_c {tuple(contact_shape)} 不一致"
        )
    if contact_shape[-1] != 1:
        raise ValueError(f"contact_temp_c 最后一维必须为 1，得到 {tuple(contact_shape)}")

    query_shape = supervision.field_query_temp_c.shape
    if supervision.field_query_xyz_base.shape != (*query_shape[:-1], 3):
        raise ValueError(
            f"field_query_xyz_base形状{tuple(supervision.field_query_xyz_base.shape)}"
            f"与温度{tuple(query_shape)}不匹配"
        )
    if supervision.field_query_mask.shape != query_shape[:-1]:
        raise ValueError(
            f"field_query_mask形状{tuple(supervision.field_query_mask.shape)}与温度{tuple(query_shape)}不匹配"
        )
    if query_shape[-1] != 1:
        raise ValueError(f"field_query_temp_c最后一维必须为1，得到{tuple(query_shape)}")

    strategy = supervision.strategy_type
    if strategy.ndim > 1:
        raise ValueError(f"strategy_type 必须是一维 [B] 或标量，得到 {tuple(strategy.shape)}")
    if bool(torch.any((strategy < SKILL_PROBE) | (strategy > SKILL_ABORT))):
        raise ValueError("strategy_type 必须是 0=PROBE / 1=GRASP / 2=RETRACT / 3=ABORT")


__all__ = [
    "SKILL_GRASP",
    "SKILL_PROBE",
    "SKILL_RETRACT",
    "SKILL_ABORT",
    "SUPERVISION_FIELDS",
    "assign_region_temperatures",
    "build_thermal_supervision",
]

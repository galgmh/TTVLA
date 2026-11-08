"""夹爪指尖的触觉传感器（Genesis 1.4.1）。

构建方式与已核实的原生路径一致：先构造 ``gs.sensors.<Class>`` 选项对象，
再在 ``scene.build()`` **之前** ``scene.add_sensor(...)``。各传感器的读数
形状（对照当前引擎）：

    ElastomerTaxel -> 位移 (N[,H], *L, 3)
    Contact        -> bool  (N[,H], 1)
    ContactForce   -> force (N[,H], 3)

``history_length`` 取控制抽减数，于是每个策略步读一次就包含该步内所有
仿真子步；记录前用中值（或末值）压掉 H 轴。

本任务只用两种触觉信号：

* ``elastomer`` —— 指尖垫切平面内的 3 维标记位移场，用于估计接触面积 M；
* ``link_bool / link_force`` —— 手指 link 级的接触判定与接触力（世界系
  接触力由 link 朝向旋转得到）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from sim.sensors.finger_layout import FingerLayout

# ---------------------------------------------------------------------------
# 触觉参数：唯一真源。YAML 只选模态，其余都在这里，避免传感器定义与数据
# 契约各自漂移。本任务记录两种信号：
#
#   elastomer -> 指尖垫切平面内的 3 维位移场（用于估计接触面积 M）
#   link_*    -> 每个手指 link 的接触判定与接触力（见 build_link_contact_sensors）
#
# ``PROBE_GRID["elastomer"]`` 是 16x16 的密集标记场。
# ---------------------------------------------------------------------------

PROBE_GRID: dict[str, tuple[int, int]] = {"elastomer": (16, 16)}
PROBE_OFFSET_M = 5e-4
PAD_TOL_M = 1.5e-3
PROBE_RADIUS_M = 0.004
# 每个被跟踪 link 的表面采样点数：引擎的 n_sample_points 是*总量*，按 link
# 均分；传入逐 link 的列表才能保持每个 link 的采样密度不随 link 数变化。
PROXIMITY_SAMPLE_POINTS = 500
AS_GRID = True
# HydroShear 长度尺度（1/m^2）：5e4 -> 法向鼓包约 4.5 mm，2e4 -> 剪切扩散
# 约 7 mm。不要把 compressibility 设成 < 1 或 elastomer_thickness > 0：
# 那会切到会崩溃的 FFT 路径。
ELASTOMER_PARAMS: dict[str, float] = {"lambda_d": 50000.0, "lambda_s": 20000.0}
CONTACT_LINKS = True
TEMPORAL_REDUCTION = "median"

# 需要非空 track_link_idx 的模态（点云类传感器的力沿指尖垫法向）。
_NEEDS_TRACK_LINK = frozenset({"elastomer"})


def grid_for(modality: str) -> tuple[int, int]:
    """某模态的 (ny, nx) 探针网格。"""
    try:
        return PROBE_GRID[modality]
    except KeyError:
        raise ValueError(f"未知触觉模态 {modality!r}；可用：{sorted(PROBE_GRID)}") from None


@dataclass
class TactileHandle:
    """一个活着的触觉传感器及其读取元数据。"""

    name: str  # 例如 "tactile_elastomer_panda_leftfinger"
    modality: str
    link_name: str
    sensor: Any
    history_length: int
    grid_shape: tuple[int, int] | None = None


def _nested(arr: np.ndarray):
    """数组 -> 嵌套元组（选项字段由 pydantic 校验）。"""
    if arr.ndim == 1:
        return tuple(float(v) for v in arr)
    return tuple(_nested(a) for a in arr)


def _probe_field(values: np.ndarray, grid: tuple[int, int] | None):
    """探针位置/法向 -> 嵌套元组；给 ``grid`` 时按 (ny,nx,3) 网格组织。"""
    arr = np.asarray(values, dtype=np.float64)
    if grid is not None:
        ny, nx = grid
        arr = arr.reshape(ny, nx, 3)
    return _nested(arr)


def _build_options(
    modality: str,
    *,
    entity_idx: int,
    link_idx_local: int,
    probe_pos,
    probe_radius: float,
    history_length: int,
    track_link_idx: tuple[int, ...],
    probe_normal=None,
    extra: dict[str, Any],
):
    """按模态构造选项对象（本任务只用到 elastomer）。"""
    import genesis as gs

    kwargs: dict[str, Any] = dict(
        entity_idx=entity_idx,
        link_idx_local=link_idx_local,
        probe_local_pos=probe_pos,
        probe_radius=probe_radius,
        history_length=history_length,
        **extra,
    )
    if modality in _NEEDS_TRACK_LINK:
        if not track_link_idx:
            raise ValueError(f"触觉模态 {modality!r} 需要 track_link_idx（全局 link 下标）")
        kwargs["track_link_idx"] = tuple(int(i) for i in track_link_idx)
        # 不传的话引擎默认 (0,0,1)，在 Panda 指尖上并不是垫面法向。
        kwargs["probe_local_normal"] = probe_normal

    if modality == "elastomer":
        return gs.sensors.ElastomerTaxel(**kwargs)
    raise ValueError(f"未知触觉模态：{modality!r}")


def build_finger_sensors(
    scene,
    *,
    entity,
    layouts_by_modality: dict[str, list[FingerLayout]],
    modalities: Sequence[str],
    decimation: int,
    extra_params: dict[str, dict[str, Any]] | None = None,
    track_link_idx: tuple[int, ...] = (),
) -> list[TactileHandle]:
    """每个（模态, 指尖 link）建一个传感器，返回活句柄。

    密度按模态区分（``grid_for``）：密集的凝胶标记场不应强迫 force 用同样
    的探针数；``layouts_by_modality`` 把每个模态映射到按*它自己*的网格推导
    出的指尖布局。

    必须在机器人实体加入之后、``scene.build()`` 之前调用。
    """
    extra_params = extra_params or {}
    for modality in modalities:
        if modality not in layouts_by_modality or not layouts_by_modality[modality]:
            raise ValueError(
                f"触觉模态 {modality!r} 没有指尖布局；"
                "scene.py 必须为每个不同探针网格各推导一套布局"
            )
    handles: list[TactileHandle] = []
    for modality in modalities:
        grid = grid_for(modality)
        for layout in layouts_by_modality[modality]:
            link = entity.get_link(layout.link_name)
            field_grid = grid if AS_GRID else None
            opts = _build_options(
                modality,
                entity_idx=entity.idx,
                link_idx_local=link.idx_local,
                probe_pos=_probe_field(layout.probe_local_pos, field_grid),
                probe_radius=PROBE_RADIUS_M,
                history_length=decimation,
                track_link_idx=track_link_idx,
                probe_normal=_probe_field(layout.probe_local_normal, field_grid),
                extra=dict(extra_params.get(modality, {})),
            )
            sensor = scene.add_sensor(opts)
            handles.append(
                TactileHandle(
                    name=f"tactile_{modality}_{layout.link_name}",
                    modality=modality,
                    link_name=layout.link_name,
                    sensor=sensor,
                    history_length=decimation,
                    grid_shape=field_grid,
                )
            )
    # read_all() 用句柄名做键，重名会静默交换两个模态的张量。
    if len({h.name for h in handles}) != len(handles):
        raise ValueError(f"触觉传感器重名：{[h.name for h in handles]}")
    return handles


def build_link_contact_sensors(scene, *, entity, link_names: Sequence[str]) -> list[TactileHandle]:
    """每个 link 一个 Contact（bool）+ 一个 ContactForce（N,3）传感器。"""
    import genesis as gs

    handles: list[TactileHandle] = []
    for name in link_names:
        link = entity.get_link(name)
        for modality, cls in (("link_bool", gs.sensors.Contact), ("link_force", gs.sensors.ContactForce)):
            sensor = scene.add_sensor(cls(entity_idx=entity.idx, link_idx_local=link.idx_local))
            handles.append(
                TactileHandle(
                    name=f"tactile_{modality}_{name}",
                    modality=modality,
                    link_name=name,
                    sensor=sensor,
                    history_length=0,
                )
            )
    return handles


# ------------------------------------------------------------------------- 读取


def _extract(data: Any, modality: str):
    """从传感器读数里取出相关张量。"""
    if modality == "elastomer":
        return data
    if isinstance(data, tuple):
        return data[0]  # Contact / ContactForce 的读数是 NamedTuple，取第一项
    return data


def temporal_reduce(tensor, history_length: int, mode: str = "median"):
    """压掉步内历史轴（第 1 轴），输入 (N, H, *S)。"""
    if history_length <= 0 or tensor.ndim < 3 or tensor.shape[1] != history_length:
        return tensor
    if mode == "median":
        return tensor.float().median(dim=1).values
    if mode == "last":
        return tensor[:, -1]
    raise ValueError(f"未知时间归约方式：{mode!r}")


def read_all(
    handles: Sequence[TactileHandle],
    *,
    envs_idx: Any = None,
    temporal_reduction: str = TEMPORAL_REDUCTION,
) -> dict[str, Any]:
    """一次性读取所有句柄：名字 -> (N[,P[,F]]) 张量（link 坐标系）。"""
    out: dict[str, Any] = {}
    for handle in handles:
        tensor = _extract(handle.sensor.read(envs_idx=envs_idx), handle.modality)
        tensor = temporal_reduce(tensor, handle.history_length, temporal_reduction)
        out[handle.name] = tensor
    return out


__all__ = [
    "AS_GRID",
    "CONTACT_LINKS",
    "ELASTOMER_PARAMS",
    "PAD_TOL_M",
    "PROBE_GRID",
    "PROBE_OFFSET_M",
    "PROBE_RADIUS_M",
    "PROXIMITY_SAMPLE_POINTS",
    "TEMPORAL_REDUCTION",
    "TactileHandle",
    "build_finger_sensors",
    "build_link_contact_sensors",
    "grid_for",
    "read_all",
    "temporal_reduce",
]

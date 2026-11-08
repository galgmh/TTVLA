"""指尖温度传感器（TemperatureGrid）的组装与读取。

本模块只负责**指尖**温度传感器：物体侧不用 TemperatureGrid，杯子四段
各是一个 link，温度存在共享表 ``link_temps`` 里（见
:mod:`sim.thermal.link_temperature`）。

YAML 只给网格尺寸、环境温度和名义非理想性四元组；下面这些实现常量是
项目级物理约定，不对外暴露：

* ``CONVECTION_COEFFICIENT = 0``：引擎的对流/辐射项把每平方米热流除以
  (ρ·cp·体素体积)，缺少面积因子，在小体素下会过冷约 1e4 倍。配
  ``emissivity = 0`` 后，一集内除接触导热外绝热——这正是本任务要的通道。
* ``SIMULATE_ALL_LINK_TEMPERATURES = True``：共享元数据一旦为真就粘住，
  场景里每个温度传感器都传同一个值；只有这样才能让杯子各 link 的温度
  互相耦合（``T_other`` 跟随对方 link 的实际温度）。
* ``CONTACT_DEPTH_WEIGHT = 0``：关闭引擎的"穿透面积"兜底项，否则毫米级
  穿透会压过真实接触面积，接触更新发散。
* ``CONTACT_AREA_POLICY = "pad_face"``：由 :mod:`sim.compat.sensor_compat`
  把指尖接触面积改写为 A_face/sqrt(n)，让一次抓取的两个手指在 2 点/3 点
  接触阶段读到同样的接触热流。要求 ``CONTACT_DEPTH_WEIGHT == 0``。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch


# ---------------------------------------------------------------- 实现常量
CONVECTION_COEFFICIENT = 0.0
SIMULATE_ALL_LINK_TEMPERATURES = True
CONTACT_DEPTH_WEIGHT = 0.0
CONTACT_AREA_POLICY = "pad_face"


@dataclass
class TemperatureHandle:
    """一个活着的 TemperatureGrid 传感器及其元数据。"""

    name: str  # 例如 "temperature_finger_panda_leftfinger"
    kind: str  # 本任务只用于 "finger"
    link_name: str
    sensor: Any
    grid_size: tuple[int, int, int]

    @property
    def num_cells(self) -> int:
        return int(self.grid_size[0] * self.grid_size[1] * self.grid_size[2])


def _shared_kwargs(ambient_temp_c: float) -> dict[str, Any]:
    """场景里每个 TemperatureGrid 共用的选项（共享元数据"后设者胜"）。"""
    return dict(
        ambient_temperature=float(ambient_temp_c),
        convection_coefficient=CONVECTION_COEFFICIENT,
        simulate_all_link_temperatures=SIMULATE_ALL_LINK_TEMPERATURES,
        contact_depth_weight=CONTACT_DEPTH_WEIGHT,
    )


def _imperfection_kwargs(nominal: dict[str, float]) -> dict[str, Any]:
    """测量通道的非理想性；三个都是按传感器下标存储的官方选项。"""
    return dict(
        sensor_time_constant=float(nominal["rc_s"]),
        noise=float(nominal["noise_c"]),
        resolution=float(nominal["quantization_c"]),
    )


def build_finger_temperature_sensors(
    scene,
    *,
    entity,
    finger_link_names: Sequence[str],
    grid_size: Sequence[int],
    ambient_temp_c: float,
    nominal: dict[str, float],
    properties_dict: dict[int, Any],
) -> list[TemperatureHandle]:
    """每个指尖 link 建一个 TemperatureGrid，顺序与 ``finger_link_names`` 一致。

    必须是 ``scene.build()`` 之前调用。指尖网格固定为 (1,1,1) 的集总单元：
    引擎按 link 世界 AABB 切网格，多单元时某一轴会塌缩到微米级（旧的
    4x4 指尖"网格"实际是 4 条 x 向窄带），集总单元没有这个问题。
    """
    import genesis as gs

    grid = tuple(int(v) for v in grid_size)
    handles: list[TemperatureHandle] = []
    for link_name in finger_link_names:
        link = entity.get_link(link_name)
        opts = gs.sensors.TemperatureGrid(
            entity_idx=entity.idx,
            link_idx_local=int(link.idx_local),
            grid_size=grid,
            properties_dict=properties_dict,
            **_shared_kwargs(ambient_temp_c),
            **_imperfection_kwargs(nominal),
        )
        sensor = scene.add_sensor(opts)
        handles.append(
            TemperatureHandle(
                name=f"temperature_finger_{link_name}",
                kind="finger",
                link_name=link_name,
                sensor=sensor,
                grid_size=grid,
            )
        )
    return handles


def seed_temperature(sensor, temperatures_c, *, envs_idx=None) -> None:
    """把一个 TemperatureGrid 直接初始化到本集的温度（没有公共 setter）。

    物理积分器演化的就是这个真值缓冲。顺序：``scene.reset()`` -> 播种 ->
    ``scene.step()``。``temperatures_c`` 是标量或长度 ``n_envs`` 的序列。

    引擎的 ``SensorManager.reset()`` 会把测量通道的时间环和当前值缓存都
    清零，而 ``TemperatureGridSensor.reset()`` 只重置真值缓冲。这里把
    测量通道的这两层也填成本集温度：读数从第一帧起就是播种值，记录里
    不会出现传感器从室温爬升到播种值的过渡。
    """
    manager = getattr(sensor, "_manager", None)
    if manager is None or not (
        hasattr(manager, "_ground_truth_intermediate_cache") and hasattr(manager, "_intermediate_cache")
    ):
        raise RuntimeError(
            "TemperatureGrid 内部结构变了（缺少真值/测量中间缓存）；"
            "请对照当前 Genesis 版本重新核对"
        )
    cls = type(sensor)
    dtype = cls._get_intermediate_dtype()
    cls_slice = manager._cache_slices_by_type[cls]
    gt = manager._ground_truth_intermediate_cache[dtype]  # (总列数, B)

    size = int(sensor._cache_size)
    start = int(cls_slice.start) + int(sensor._cache_offset)
    assert size == int(sensor._shared_metadata.cache_sizes[sensor._idx]), "缓存切片不一致"

    if envs_idx is None:
        envs = torch.arange(gt.shape[1], device=gt.device)
    else:
        envs = manager._sim._scene._sanitize_envs_idx(envs_idx)

    vals = torch.as_tensor(temperatures_c, dtype=gt.dtype, device=gt.device).reshape(-1)
    if vals.numel() == 1:
        vals = vals.expand(envs.numel())
    if vals.numel() != envs.numel():
        raise ValueError(f"temperatures_c 有 {vals.numel()} 个值，需要 {envs.numel()} 个")

    gt[start : start + size, envs] = vals.view(1, -1).expand(size, -1)

    md = sensor._shared_metadata
    if md.link_temps.numel() > 0:
        md.link_temps[envs, md.links_idx[sensor._idx]] = vals

    # 测量通道的两层状态都填成本集温度：
    # * 时间环是 RC 滤波的历史（下一步的 previous 从这里取）；
    # * 中间缓存是 read() 直接返回的当前值。
    # 这样复位后立刻读也是播种值，记录里不会出现从室温收敛的过渡段。
    rings = getattr(manager, "_measured_timeline_ring", None)
    if rings is None:
        raise RuntimeError("SensorManager 缺少 _measured_timeline_ring；测量通道无法初始化")
    ring = rings.get(dtype)
    if ring is None:
        raise RuntimeError(f"dtype {dtype} 没有测量通道时间环；TemperatureGrid 的阻尼滤波无处可写")
    ring.buffer[:, envs, start : start + size] = vals.view(1, -1, 1).expand(
        ring.buffer.shape[0], -1, size
    )
    measured = manager._intermediate_cache[dtype][:, start : start + size]
    measured[envs] = vals.view(-1, 1).expand(-1, size)


def read_temperature(
    handles: Sequence[TemperatureHandle],
    *,
    ground_truth: bool = False,
) -> dict[str, torch.Tensor]:
    """读每个指尖网格：名字 -> (n_envs, *grid_size) 摄氏度张量。

    默认读*测量*通道（带 RC / 噪声 / 量化）；``ground_truth=True`` 读未受
    非理想性影响的值，仅用于自检。
    """
    out: dict[str, torch.Tensor] = {}
    for handle in handles:
        if ground_truth:
            out[handle.name] = handle.sensor.read_ground_truth()
        else:
            out[handle.name] = handle.sensor.read()
    return out


__all__ = [
    "CONTACT_AREA_POLICY",
    "CONVECTION_COEFFICIENT",
    "CONTACT_DEPTH_WEIGHT",
    "SIMULATE_ALL_LINK_TEMPERATURES",
    "TemperatureHandle",
    "build_finger_temperature_sensors",
    "read_temperature",
    "seed_temperature",
]

"""热区 link 温度与指尖初始温度的读写。

引擎把"每个 link 一个集总温度"存在共享表 ``link_temps`` 里（形状
``(n_envs, n_links)``）。温度传感器自身的 ``reset()`` 会把该表写回材料
基准温度，因此**每集播种必须发生在 ``scene.reset()`` 之后、下一次
``step()`` 之前**，这个顺序由 :meth:`sim.scene.SceneBundle.reset_episode`
保证。

指尖传感器是引擎默认的 (1,1,1) 集总单元，其 ``read()`` 走测量通道，带
传感器非理想性（RC 滞后、噪声、量化）。每集随机化这几个参数在这里落地：
公共 setter 只能改噪声/偏置/量化，RC 需要直接写共享元数据里的
``sensor_time_const``（按传感器下标）。
"""

from __future__ import annotations

from typing import Any, Sequence

import torch

from sim.regions import ThermalRegionRuntime


def _envs(handle, envs_idx) -> torch.Tensor:
    """把可选的环境下标规范成张量；None 表示全部环境。"""
    table = handle.sensor._shared_metadata.link_temps
    if envs_idx is None:
        return torch.arange(table.shape[0], device=table.device)
    return handle.sensor._manager._sim._scene._sanitize_envs_idx(envs_idx)


def seed_link_temperatures(
    handle: Any,
    link_indices: Sequence[int],
    temperatures_c: Sequence[float] | float,
    *,
    envs_idx=None,
) -> None:
    """播种所选 link 的集总初始温度（每个环境一份）。

    复合刚体（固定关节连起来的 URDF）在热模型里是*每个 link 一个热节点*，
    接触传热核读取的是被接触 link 的那一行（``T_other = link_temps[i_b, link]``）。
    直接写这些行，才能让每个区域有各自的起始温度。

    ``handle`` 可以是场景里任意一个温度传感器：``link_temps`` 是共享元数据，
    用哪个句柄都一样。``temperatures_c`` 是一个值（广播）或每个 link 一个。
    顺序规则：``scene.reset()`` -> 播种 -> ``scene.step()``。
    """
    md = handle.sensor._shared_metadata
    table = md.link_temps
    if table.numel() == 0:
        raise RuntimeError(
            "link_temps 为空：场景里没有 simulate_all_link_temperatures=True 的温度传感器，"
            "逐 link 播种无处可写"
        )
    idx = torch.as_tensor([int(i) for i in link_indices], dtype=torch.long, device=table.device).reshape(-1)
    vals = torch.as_tensor(temperatures_c, dtype=table.dtype, device=table.device).reshape(-1)
    if vals.numel() == 1:
        vals = vals.expand(idx.numel())
    if vals.numel() != idx.numel():
        raise ValueError(f"temperatures_c 有 {vals.numel()} 个值，需要 {idx.numel()} 个（每 link 一个）")
    envs = _envs(handle, envs_idx)
    # 高级索引得到 (n_envs, n_links) 选择；一行值沿环境维广播。
    table[envs, idx] = vals.view(1, -1)


def seed_region_temperatures(
    handle: Any,
    region_map: ThermalRegionRuntime,
    temperatures_c: Sequence[float] | dict[str, float] | float,
    *,
    envs_idx=None,
) -> None:
    """按热区表播种杯子四段的初始温度。

    ``temperatures_c`` 可以是：一个标量、按 ``region_order`` 排列的序列、
    或 ``{区域名: 温度}`` 映射。缺区域会报错——静默漏播种会让该段保留
    上一集（或基准）的温度。
    """
    indices = [int(i) for i in region_map.global_link_indices]
    if isinstance(temperatures_c, dict):
        missing = set(region_map.order) - set(temperatures_c)
        if missing:
            raise ValueError(f"区域温度缺少 {sorted(missing)}")
        values = [float(temperatures_c[name]) for name in region_map.order]
    else:
        values = temperatures_c
    seed_link_temperatures(handle, indices, values, envs_idx=envs_idx)


def seed_fingertip_temperatures(
    handles: Sequence[Any],
    temperatures_c: Sequence[float] | float,
    *,
    envs_idx=None,
) -> None:
    """播种每个指尖传感器的初始温度（左、右各一个值）。"""
    from sim.sensors.temperature import seed_temperature

    values = (
        [float(v) for v in temperatures_c]
        if isinstance(temperatures_c, (list, tuple))
        else float(temperatures_c)
    )
    if isinstance(values, list) and len(values) != len(handles):
        raise ValueError(f"指尖温度有 {len(values)} 个值，需要 {len(handles)} 个")
    for i, handle in enumerate(handles):
        value = values[i] if isinstance(values, list) else values
        seed_temperature(handle.sensor, value, envs_idx=envs_idx)


def apply_sensor_variation(
    handles: Sequence[Any],
    *,
    rc_s: float,
    noise_c: float,
    bias_c: float = 0.0,
    quantization_c: float = 0.0,
) -> None:
    """把本集的传感器非理想性写进活着的传感器实例。

    必须在 ``scene.build()`` 之后调用（setter 带 ``assert_built``）。RC 没有
    公共 setter：共享元数据按传感器下标保存 ``sensor_time_const``，直接改
    那一格即可，下一次 ``read()`` 生效。每集复位都会重新写一遍。
    """
    for handle in handles:
        sensor = handle.sensor
        sensor.set_noise(float(noise_c))
        sensor.set_bias(float(bias_c))
        sensor.set_resolution(float(quantization_c))
        md = sensor._shared_metadata
        if md.sensor_time_const.numel() > 0:
            md.sensor_time_const[int(sensor._idx)] = float(rc_s)


def read_link_temperatures(handle: Any) -> torch.Tensor | None:
    """共享的逐 link 温度表 (n_envs, n_links)；未启用时返回 None。"""
    temps = handle.sensor.link_temperatures
    return temps if temps.numel() > 0 else None


def read_region_temperatures(
    handle: Any, region_map: ThermalRegionRuntime
) -> torch.Tensor:
    """按区域顺序取杯子四段的温度，形状 (n_envs, 4)。"""
    table = read_link_temperatures(handle)
    if table is None:
        raise RuntimeError("link_temps 为空，无法读取区域温度")
    idx = torch.as_tensor(
        [int(i) for i in region_map.global_link_indices], dtype=torch.long, device=table.device
    )
    return table[:, idx]


__all__ = [
    "apply_sensor_variation",
    "read_link_temperatures",
    "read_region_temperatures",
    "seed_fingertip_temperatures",
    "seed_link_temperatures",
    "seed_region_temperatures",
]

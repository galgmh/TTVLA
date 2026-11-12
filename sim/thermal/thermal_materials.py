"""接触传热所需的材料属性表。

引擎的 ``TemperatureGrid`` 用 ``properties_dict`` 描述"每个 link 是什么
材料"：键是场景级 link 下标，``-1`` 是未列出 link 的兜底项。接触传热核
按两侧材料的 ``conductivity`` 折算等效导热（k_eff = 2·k1·k2/(k1+k2)），
因此指尖读数天然带有材料对比——本任务的标签是"接触段温度"，材料只是
背景物理，不进入模型输入。

约定：

* 指尖 link —— 硅胶参数（``FINGER_PROPERTIES``）；
* 杯子四段 —— 物体配置里 ``material`` 指向的那组参数；
* 其余（桌面、平台、机器人手臂）—— 环境参数（``ENVIRONMENT_PROPERTIES``），
  它们与杯子的接触几乎不发生，但兜底项必须存在，否则未列出的 link 完全
  不参与传热。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

# 指尖硅胶。k=0.25 是真实硅胶量级；它同时决定引擎显式接触更新的增益，
# 必须保持稳定（每个子步 c = dt·k_eff·A²/(V²·ρ·cp) < 0.5）。
FINGER_PROPERTIES: dict[str, float] = {
    "conductivity": 0.25,
    "density": 1100.0,
    "specific_heat": 1400.0,
    "emissivity": 0.0,
}

# 桌面 / 平台 / 机械臂的兜底材料。
ENVIRONMENT_PROPERTIES: dict[str, float] = {
    "conductivity": 0.15,
    "density": 700.0,
    "specific_heat": 1600.0,
    "emissivity": 0.0,
}


def properties_from_material(
    props: Mapping[str, float],
    *,
    base_temperature_c: float | None = None,
):
    """把配置里的材料字典转成 ``TemperatureProperties``。"""
    from genesis.options.sensors import TemperatureProperties

    kwargs = {k: float(v) for k, v in props.items() if k in TemperatureProperties._fields}
    if base_temperature_c is not None:
        kwargs["base_temperature"] = float(base_temperature_c)
    return TemperatureProperties(**kwargs)


def make_properties_dict(
    links: Mapping[int, Mapping[str, float] | Any],
    *,
    default: Mapping[str, float] | Any | None = None,
) -> dict[int, Any]:
    """组装 properties_dict（场景级 link 下标 -> TemperatureProperties）。

    ``-1`` 键是未列出 link 的兜底项；不给则这些 link 完全不参与接触传热。
    """
    out: dict[int, Any] = {}
    for idx, props in links.items():
        out[int(idx)] = props if hasattr(props, "_fields") else properties_from_material(props)
    if default is not None:
        out[-1] = default if hasattr(default, "_fields") else properties_from_material(default)
    return out


def _global_link_index(link: Any) -> int:
    """link 的场景级下标（整数）。"""
    idx = getattr(link, "idx")
    if hasattr(idx, "detach"):
        idx = idx.detach().cpu()
    flat = [int(v) for v in (idx.reshape(-1) if hasattr(idx, "reshape") else [idx])]
    if len(flat) != 1:
        raise ValueError(f"link {getattr(link, 'name', '?')} 的 idx 不是单值：{flat}")
    return flat[0]


def build_properties_dict(
    *,
    object_cfg: dict[str, Any],
    sensor_cfg: dict[str, Any],
    finger_links: Sequence[Any],
    cup_link_indices: Sequence[int],
) -> dict[int, Any]:
    """按"指尖 / 杯子 / 环境"三类组装完整的属性表。

    ``finger_links`` 是机器人实体里的指尖 link 对象（取 ``.idx`` 全局下标），
    ``cup_link_indices`` 是热区表给出的杯子四段全局下标。
    """
    ambient = float(sensor_cfg["ambient_temp_c"])
    material = str(object_cfg["material"])
    props = object_cfg["material_properties"]
    if material not in props:
        raise ValueError(f"material = {material!r} 不在 material_properties 中")

    links: dict[int, dict[str, float]] = {}
    for link in finger_links:
        links[_global_link_index(link)] = dict(FINGER_PROPERTIES)
    region_order = tuple(str(name) for name in object_cfg["region_order"])
    if len(region_order) != len(cup_link_indices):
        raise ValueError(
            f"region_order 有 {len(region_order)} 项，cup_link_indices 有 {len(cup_link_indices)} 项"
        )
    link_materials = object_cfg.get("link_materials", {})
    for region_name, idx in zip(region_order, cup_link_indices):
        material_name = str(link_materials.get(region_name, material))
        if material_name not in props:
            raise ValueError(f"区域 {region_name!r} 的材料 {material_name!r} 未定义")
        links[int(idx)] = dict(props[material_name])

    return make_properties_dict(
        {
            idx: properties_from_material(values, base_temperature_c=ambient)
            for idx, values in links.items()
        },
        default=properties_from_material(ENVIRONMENT_PROPERTIES, base_temperature_c=ambient),
    )


__all__ = [
    "ENVIRONMENT_PROPERTIES",
    "FINGER_PROPERTIES",
    "build_properties_dict",
    "make_properties_dict",
    "properties_from_material",
]

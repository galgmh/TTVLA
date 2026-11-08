"""传感器包的懒导出。

``import sim.sensors`` 不应拉起 torch / Genesis：只依赖 numpy 的
``finger_layout`` 要能脱离引擎栈单独导入和测试。PEP 562 的模块级
``__getattr__`` 只在真正取某个名字时才加载对应子模块。
"""

from __future__ import annotations

from importlib import import_module

# 公开名 -> (子模块, 属性)
_EXPORTS: dict[str, tuple[str, str]] = {
    # finger_layout.py（纯 XML + numpy）
    "FingerLayout": (".finger_layout", "FingerLayout"),
    "derive_finger_layout": (".finger_layout", "derive_finger_layout"),
    "parse_child_joints": (".finger_layout", "parse_child_joints"),
    "parse_link_boxes": (".finger_layout", "parse_link_boxes"),
    "parse_link_meshes": (".finger_layout", "parse_link_meshes"),
    "rpy_to_matrix": (".finger_layout", "rpy_to_matrix"),
    # tactile.py（模块级只用 numpy；torch 只在读取时出现）
    "TactileHandle": (".tactile", "TactileHandle"),
    "build_finger_sensors": (".tactile", "build_finger_sensors"),
    "build_link_contact_sensors": (".tactile", "build_link_contact_sensors"),
    "grid_for": (".tactile", "grid_for"),
    # temperature.py（模块级导入 torch；按需加载）
    "TemperatureHandle": (".temperature", "TemperatureHandle"),
    "build_finger_temperature_sensors": (".temperature", "build_finger_temperature_sensors"),
    "read_temperature": (".temperature", "read_temperature"),
    "seed_temperature": (".temperature", "seed_temperature"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attr = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"模块 {__name__!r} 没有属性 {name!r}") from None
    value = getattr(import_module(module_name, __name__), attr)
    globals()[name] = value  # 缓存；后续查找不再走 __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(__all__)

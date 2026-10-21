"""Lazy public exports for the sensor package.

Importing :mod:`sim.sensors` must not pull torch or Genesis: the pure-numpy
tier (config, tools, events) has to stay importable without the engine stack.
PEP 562 module ``__getattr__`` loads a submodule only when one of its names is
actually requested, so ``from sim.sensors import read_tactile`` still works
while ``import sim.sensors`` stays light.
"""

from __future__ import annotations

from importlib import import_module

# public name -> (module, attribute)
_EXPORTS: dict[str, tuple[str, str]] = {
    # finger_layout.py (pure XML + numpy)
    "FingerLayout": (".finger_layout", "FingerLayout"),
    "derive_finger_layout": (".finger_layout", "derive_finger_layout"),
    "grid_tag": (".finger_layout", "grid_tag"),
    "layouts_to_hdf5": (".finger_layout", "layouts_to_hdf5"),
    "layouts_to_hdf5_multi": (".finger_layout", "layouts_to_hdf5_multi"),
    "summarize": (".finger_layout", "summarize"),
    # tactile.py (numpy at module level; torch only inside a force_torque read)
    "TactileHandle": (".tactile", "TactileHandle"),
    "build_finger_sensors": (".tactile", "build_finger_sensors"),
    "build_link_contact_sensors": (".tactile", "build_link_contact_sensors"),
    "read_tactile": (".tactile", "read_all"),
    "rotate_to_world": (".tactile", "rotate_to_world"),
    "temporal_reduce": (".tactile", "temporal_reduce"),
    # format-1.6 derived signals: one parameter source, pure numpy
    "grid_for": (".tactile", "grid_for"),
    "primary_modality": (".tactile", "primary_modality"),
    "channel_format": (".tactile", "channel_format"),
    "pad_tangent_basis": (".tactile", "pad_tangent_basis"),
    "force_6d": (".tactile", "force_6d"),
    "marker_2d": (".tactile", "marker_2d"),
    # temperature.py (torch at module level; loaded on demand only)
    "TemperatureHandle": (".temperature", "TemperatureHandle"),
    "build_finger_temperature_sensors": (".temperature", "build_finger_temperature_sensors"),
    "build_object_temperature_sensor": (".temperature", "build_object_temperature_sensor"),
    "build_temperature_sensor": (".temperature", "build_temperature_sensor"),
    "make_properties_dict": (".temperature", "make_properties_dict"),
    "properties_from_material": (".temperature", "properties_from_material"),
    "read_link_temperatures": (".temperature", "read_link_temperatures"),
    "read_temperature": (".temperature", "read_temperature"),
    "seed_temperature": (".temperature", "seed_temperature"),
    "seed_temperature_all": (".temperature", "seed_all"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attr = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name, __name__), attr)
    globals()[name] = value  # cache; subsequent lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(__all__)

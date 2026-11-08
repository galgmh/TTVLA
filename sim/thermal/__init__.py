"""热模型：逐 link 温度播种与材料属性表。"""

from sim.thermal.link_temperature import (
    apply_sensor_variation,
    read_link_temperatures,
    read_region_temperatures,
    seed_fingertip_temperatures,
    seed_link_temperatures,
    seed_region_temperatures,
)
from sim.thermal.thermal_materials import (
    ENVIRONMENT_PROPERTIES,
    FINGER_PROPERTIES,
    build_properties_dict,
    make_properties_dict,
    properties_from_material,
)

__all__ = [
    "ENVIRONMENT_PROPERTIES",
    "FINGER_PROPERTIES",
    "apply_sensor_variation",
    "build_properties_dict",
    "make_properties_dict",
    "properties_from_material",
    "read_link_temperatures",
    "read_region_temperatures",
    "seed_fingertip_temperatures",
    "seed_link_temperatures",
    "seed_region_temperatures",
]

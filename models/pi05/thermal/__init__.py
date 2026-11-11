"""Thermal-π0.5 模型层：三个核心模块与 π0.5 接入。

1. ``ThermalContactEncoder``：把每次短时接触的温度/力觉序列编码为局部热觉 token；
2. ``ThermalSpatialMemory``：把任意数量的历史接触投影到物体表面候选点，形成空间热场；
3. ``ActiveStrategyModule``：根据语言、视觉、机器人状态和空间热场选择继续探测还是
   抓取，并产生可微的目标 token。

``IntegratedThermalEncoder`` 把三者串起来，并压成 M+1 个 π0.5 prefix token。
训练/数据侧工具（历史缓存、候选点、标签、数据变换、闭环驱动、分阶段冻结、训练
wrapper）不在模型层，放在 ``training/pi05/thermal/``。
"""

from models.pi05.thermal.config import ThermalPi0Config
from models.pi05.thermal.contact_encoder import ThermalContactEncoder
from models.pi05.thermal.integrated_encoder import IntegratedThermalEncoder
from models.pi05.thermal.outputs import ContactEncoderOutput
from models.pi05.thermal.outputs import IntegratedThermalOutput
from models.pi05.thermal.outputs import SpatialMemoryOutput
from models.pi05.thermal.outputs import StrategyOutput
from models.pi05.thermal.outputs import ThermalSupervision
from models.pi05.thermal.point_encoder import SurfacePointEncoder
from models.pi05.thermal.spatial_memory import ContactTokenBuilder
from models.pi05.thermal.spatial_memory import ThermalSpatialMemory
from models.pi05.thermal.strategy import ActiveStrategyModule

__all__ = [
    "ActiveStrategyModule",
    "ContactEncoderOutput",
    "ContactTokenBuilder",
    "IntegratedThermalEncoder",
    "IntegratedThermalOutput",
    "SpatialMemoryOutput",
    "StrategyOutput",
    "SurfacePointEncoder",
    "ThermalContactEncoder",
    "ThermalPi0Config",
    "ThermalSpatialMemory",
    "ThermalSupervision",
]

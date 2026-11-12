"""修改版 Thermal-π0.5 模型层。

1. ``ThermalContactEncoder``：把每次短时接触的温度/力觉序列编码为局部热觉 token；
2. ``ThermalSpatialMemory``：把已完成接触集合压缩成空间记忆，不接收候选点；
3. ``GlobalSkillAuxHead``：只提供全局技能辅助监督，不产生触点或goal token；
4. ``ThermalPrefixAdapter``：只把物理空间记忆映射为 M 个 PaliGemma token。

``IntegratedThermalEncoder`` 把模块串起来，并压成 M 个 π0.5 prefix token。
训练/数据侧工具（标签、数据变换、闭环驱动、分阶段冻结、训练
wrapper）不在模型层，放在 ``training/pi05/thermal/``。
"""

from models.pi05.thermal.config import ThermalPi0Config
from models.pi05.thermal.contact_encoder import ThermalContactEncoder
from models.pi05.thermal.integrated_encoder import IntegratedThermalEncoder
from models.pi05.thermal.memory_bank import ThermalMemoryBank
from models.pi05.thermal.outputs import ContactEncoderOutput
from models.pi05.thermal.outputs import FieldDecoderOutput
from models.pi05.thermal.outputs import IntegratedThermalOutput
from models.pi05.thermal.outputs import SpatialMemoryOutput
from models.pi05.thermal.outputs import StrategyAuxOutput
from models.pi05.thermal.outputs import ThermalSupervision
from models.pi05.thermal.field_decoder import SpatialFieldDecoder
from models.pi05.thermal.spatial_memory import ContactTokenBuilder
from models.pi05.thermal.spatial_memory import ThermalSpatialMemory
from models.pi05.thermal.prefix_adapter import ThermalPrefixAdapter
from models.pi05.thermal.strategy_aux import GlobalSkillAuxHead

__all__ = [
    "GlobalSkillAuxHead",
    "ContactEncoderOutput",
    "ContactTokenBuilder",
    "FieldDecoderOutput",
    "IntegratedThermalEncoder",
    "IntegratedThermalOutput",
    "SpatialMemoryOutput",
    "StrategyAuxOutput",
    "ThermalPrefixAdapter",
    "ThermalMemoryBank",
    "SpatialFieldDecoder",
    "ThermalContactEncoder",
    "ThermalPi0Config",
    "ThermalSpatialMemory",
    "ThermalSupervision",
]

"""热编码器模型层：编码器、两个头、损失。

这一层只依赖 torch，不 import 仿真或数据层；训练脚本负责把加载器给出的
窗口张量喂进来。
"""

from models.losses import LossTerms, compute_losses, masked_bce, masked_huber
from models.thermal_encoder import ThermalEncoder
from models.thermal_heads import ContactValidityHead, TemperatureHead
from models.thermal_model import ThermalModel, ThermalOutput

__all__ = [
    "ContactValidityHead",
    "LossTerms",
    "TemperatureHead",
    "ThermalEncoder",
    "ThermalModel",
    "ThermalOutput",
    "compute_losses",
    "masked_bce",
    "masked_huber",
]

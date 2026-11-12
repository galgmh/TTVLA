"""热编码器模型层：共享接触编码器、温度分布/有效性输出与损失。

这一层只依赖 torch，不 import 仿真或数据层；训练脚本负责把加载器给出的
窗口张量喂进来。
"""

from models.losses import LossTerms, compute_losses, masked_bce, masked_gaussian_nll, masked_huber
from models.thermal_encoder import ThermalEncoder, load_contact_encoder_checkpoint
from models.thermal_model import ThermalModel, ThermalOutput

__all__ = [
    "LossTerms",
    "ThermalEncoder",
    "load_contact_encoder_checkpoint",
    "ThermalModel",
    "ThermalOutput",
    "compute_losses",
    "masked_bce",
    "masked_gaussian_nll",
    "masked_huber",
]

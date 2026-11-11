"""官方 π0.5 模型代码的项目内副本（纯 PyTorch，不依赖 JAX）。

来源：openpi 提交 ``215abfb217dbac7d5f1273282331b9b1866c0479``。
import 路径由 ``openpi.*`` 改写为 ``models.pi05.*``；JAX/Flax 参考实现已移除，
训练、微调与评测均走 PyTorch 路径。``thermal/`` 为 Thermal-π0.5 热觉分支
（热觉关闭时数值路径与官方 π0.5 完全相同）。

运行环境：torch 2.7.1 / transformers 4.53.2，并把
``models/pi05/transformers_replace/`` 拷入 site-packages 的 ``transformers/``。
"""

from models.pi05.observation import Observation
from models.pi05.pi0_config import Pi0Config
from models.pi05.pi0_pytorch import PI0Pytorch
from models.pi05.pi0_pytorch import load_state_dict_compat
from models.pi05.thermal import IntegratedThermalEncoder
from models.pi05.thermal import IntegratedThermalOutput
from models.pi05.thermal import ThermalPi0Config
from models.pi05.thermal import ThermalSupervision

__all__ = [
    "IntegratedThermalEncoder",
    "IntegratedThermalOutput",
    "Observation",
    "PI0Pytorch",
    "Pi0Config",
    "ThermalPi0Config",
    "ThermalSupervision",
    "load_state_dict_compat",
]

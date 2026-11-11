"""Thermal-π0.5 训练/微调/评测层（纯 PyTorch，参考 openpi `scripts/train_pytorch.py` 重写）。

这一层只依赖 `models.pi05`（模型）与项目数据，不依赖外部 openpi 安装，也不需要 JAX。
入口：
``python -m training.pi05.train --config configs/training/pi05_thermal_stage1.yaml``；
``python -m training.pi05.eval --config configs/training/pi05_thermal_stage1.yaml``。
"""

from training.pi05.config import DataConfig
from training.pi05.config import LRScheduleConfig
from training.pi05.config import OptimizerConfig
from training.pi05.config import TrainConfig
from training.pi05.config import load_train_config

__all__ = [
    "DataConfig",
    "LRScheduleConfig",
    "OptimizerConfig",
    "TrainConfig",
    "load_train_config",
]

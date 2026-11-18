"""Thermal-π0.5 训练/数据侧工具（不属于模型层）。

模型层只保留最终在线热觉编码器和 π0.5 prefix 接入；辅助头和辅助 loss
按训练阶段放在 ``spatial_pretraining`` 与本包：

- ``finetune.py``：π0.5 接入后的冻结与三组学习率；
- ``losses.py``：可选辅助监督项；
- ``labels.py``：特权标签 -> ``ThermalSupervision``；
- ``rollout.py``：闭环推理驱动；
- ``step.py``：训练 wrapper ``thermal_train_step``；
- ``transforms.py``：数据批次 -> ``Observation`` 热觉字段。
"""

from thermal_model.pi05_training.finetune import STAGES
from thermal_model.pi05_training.finetune import configure_stage
from thermal_model.pi05_training.finetune import parameter_groups
from thermal_model.pi05_training.finetune import stage_loss_keys
from thermal_model.pi05_training.labels import build_thermal_supervision
from thermal_model.pi05_training.rollout import ThermalClosedLoop
from thermal_model.pi05_training.step import thermal_train_step
from thermal_model.pi05_training.transforms import EmptyThermalObservation
from thermal_model.pi05_training.transforms import ThermalObservationTransform
from thermal_model.pi05_training.transforms import thermal_supervision_from_batch

__all__ = [
    "STAGES",
    "EmptyThermalObservation",
    "ThermalClosedLoop",
    "ThermalObservationTransform",
    "build_thermal_supervision",
    "configure_stage",
    "parameter_groups",
    "stage_loss_keys",
    "thermal_supervision_from_batch",
    "thermal_train_step",
]

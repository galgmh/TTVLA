"""Thermal-π0.5 训练/数据侧工具（不属于模型层）。

模型层 ``models/pi05/thermal/`` 只保留三个核心模块与 π0.5 接入；本包放：

- ``finetune.py``：分阶段冻结与三组学习率；
- ``history.py``：因果接触历史缓存；
- ``labels.py``：特权标签 -> ``ThermalSupervision``；
- ``rollout.py``：闭环推理驱动；
- ``step.py``：训练 wrapper ``thermal_train_step``；
- ``surfaces.py``：候选表面点组装；
- ``transforms.py``：数据批次 -> ``Observation`` 热觉字段。
"""

from training.pi05.thermal.finetune import STAGES
from training.pi05.thermal.finetune import configure_stage
from training.pi05.thermal.finetune import parameter_groups
from training.pi05.thermal.finetune import stage_loss_keys
from training.pi05.thermal.history import ContactHistoryBuffer
from training.pi05.thermal.labels import build_thermal_supervision
from training.pi05.thermal.rollout import CompletedContact
from training.pi05.thermal.rollout import ThermalClosedLoop
from training.pi05.thermal.step import thermal_train_step
from training.pi05.thermal.surfaces import SurfaceCandidates
from training.pi05.thermal.surfaces import build_surface_candidates
from training.pi05.thermal.transforms import EmptyThermalObservation
from training.pi05.thermal.transforms import ThermalObservationTransform
from training.pi05.thermal.transforms import thermal_supervision_from_batch

__all__ = [
    "STAGES",
    "CompletedContact",
    "ContactHistoryBuffer",
    "EmptyThermalObservation",
    "SurfaceCandidates",
    "ThermalClosedLoop",
    "ThermalObservationTransform",
    "build_surface_candidates",
    "build_thermal_supervision",
    "configure_stage",
    "parameter_groups",
    "stage_loss_keys",
    "thermal_supervision_from_batch",
    "thermal_train_step",
]

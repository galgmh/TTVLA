"""Thermal-π0.5 各模块的输出结构与训练监督容器。"""

from __future__ import annotations

import dataclasses

import torch


@dataclasses.dataclass
class ContactEncoderOutput:
    """当前实时滑动窗口的编码结果。"""

    latent: torch.Tensor  # [B, thermal_latent_dim]
    temp_mean_c: torch.Tensor  # [B, 1]
    temp_logvar: torch.Tensor  # [B, 1]
    valid_logit: torch.Tensor  # [B, 1]


@dataclasses.dataclass
class SpatialMemoryOutput:
    """已完成接触集合压缩后的空间记忆。"""

    spatial_latents: torch.Tensor  # [B, M, hidden_dim]


@dataclasses.dataclass
class FieldDecoderOutput:
    """训练/评测查询点的温度分布，不进入控制prefix。"""

    temp_mean_c: torch.Tensor  # [B, Q, 1]
    temp_logvar: torch.Tensor  # [B, Q, 1]


@dataclasses.dataclass
class StrategyAuxOutput:
    """策略辅助监督输出；不直接条件化 PaliGemma 或 Action Expert。"""

    skill_logits: torch.Tensor  # [B, 4]：PROBE/GRASP/RETRACT/ABORT


@dataclasses.dataclass
class IntegratedThermalOutput:
    """IntegratedThermalEncoder 一次前向的全部产物。"""

    prefix_tokens: torch.Tensor  # [B, M, paligemma_width]
    prefix_mask: torch.Tensor  # [B, M]
    contact: ContactEncoderOutput
    spatial_latents: torch.Tensor  # [B, M, hidden_dim]
    field: FieldDecoderOutput | None
    strategy: StrategyAuxOutput | None


@dataclasses.dataclass
class ThermalSupervision:
    """训练监督字段（仿真特权标签），只参与损失，不进入 Observation。"""

    contact_temp_c: torch.Tensor  # [B, 1]
    contact_valid: torch.Tensor  # [B, 1]
    field_query_xyz_base: torch.Tensor  # [B, Q, 3]，仅训练/评测decoder使用
    field_query_temp_c: torch.Tensor  # [B, Q, 1]
    field_query_mask: torch.Tensor  # [B, Q]
    strategy_type: torch.Tensor  # [B]：0=PROBE, 1=GRASP, 2=RETRACT, 3=ABORT

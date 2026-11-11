"""Thermal-π0.5 各模块的输出结构与训练监督容器。"""

from __future__ import annotations

import dataclasses

import torch


@dataclasses.dataclass
class ContactEncoderOutput:
    """单次接触窗口的编码结果（每个样本 K 次历史接触）。"""

    latent: torch.Tensor  # [B, K, thermal_latent_dim]
    temp_mean_c: torch.Tensor  # [B, K, 1]
    temp_logvar: torch.Tensor  # [B, K, 1]
    valid_logit: torch.Tensor  # [B, K, 1]


@dataclasses.dataclass
class SpatialMemoryOutput:
    """候选表面点的热场表示与压缩后的空间 token。"""

    surface_features: torch.Tensor  # [B, N, hidden_dim]
    temp_mean_c: torch.Tensor  # [B, N, 1]
    temp_logvar: torch.Tensor  # [B, N, 1]
    safe_logit: torch.Tensor  # [B, N, 1]
    spatial_latents: torch.Tensor  # [B, M, hidden_dim]


@dataclasses.dataclass
class StrategyOutput:
    """主动策略输出：技能分布 + 可微目标点选择。"""

    skill_logits: torch.Tensor  # [B, 3]：0=OTHER, 1=PROBE, 2=GRASP
    probe_logits: torch.Tensor  # [B, N]
    grasp_logits: torch.Tensor  # [B, N]
    selected_distribution: torch.Tensor  # [B, N]
    goal_feature: torch.Tensor  # [B, hidden_dim]
    selected_index: torch.Tensor | None = None  # 评测模式（masked argmax）下的目标点


@dataclasses.dataclass
class IntegratedThermalOutput:
    """IntegratedThermalEncoder 一次前向的全部产物。"""

    prefix_tokens: torch.Tensor  # [B, M+1, paligemma_width]
    prefix_mask: torch.Tensor  # [B, M+1]
    contact: ContactEncoderOutput
    field_temp_mean_c: torch.Tensor  # [B, N, 1]
    field_temp_logvar: torch.Tensor  # [B, N, 1]
    field_safe_logit: torch.Tensor  # [B, N, 1]
    strategy: StrategyOutput


@dataclasses.dataclass
class ThermalSupervision:
    """训练监督字段（仿真特权标签），只参与损失，不进入 Observation。"""

    contact_temp_c: torch.Tensor  # [B, K, 1]
    contact_valid: torch.Tensor  # [B, K, 1]
    surface_temp_c: torch.Tensor  # [B, N, 1]
    surface_temp_mask: torch.Tensor  # [B, N]
    surface_safe: torch.Tensor  # [B, N, 1]
    strategy_type: torch.Tensor  # [B]：0=OTHER, 1=PROBE, 2=GRASP
    target_surface_index: torch.Tensor  # [B]：OTHER 时为 -1

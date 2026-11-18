"""最终在线热觉模型的输出结构。"""

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
class IntegratedThermalOutput:
    """IntegratedThermalEncoder 一次前向的全部产物。"""

    prefix_tokens: torch.Tensor  # [B, M, paligemma_width]
    prefix_mask: torch.Tensor  # [B, M]
    contact: ContactEncoderOutput
    spatial_latents: torch.Tensor  # [B, M, hidden_dim]

"""Thermal-π0.5 的模块配置。

所有维度（K/L/C/N/M/D）都从这里读取，模型内部不硬编码；
``enabled=False`` 时 PI0Pytorch 不创建任何热觉模块，行为与原 π0.5 完全一致。
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class ThermalPi0Config:
    """热觉分支的可冻结配置（作为 Pi0Config.thermal 一起序列化）。"""

    # 总开关：False 时原模型行为不变（Stage 0 回归基线）。
    enabled: bool = False

    # 历史接触窗口：[B, K, L, C]。K 为历史上限，允许 0..K 个有效接触。
    history_size: int = 8
    window_size: int = 25
    signal_dim: int = 14

    # 当前候选表面点数量。
    num_surface_points: int = 64

    # 新增模块内部维度、热接触 latent 维度、压缩后送入 π0.5 的空间 token 数。
    hidden_dim: int = 256
    thermal_latent_dim: int = 128
    num_spatial_tokens: int = 8

    # 主动策略：训练后期可切到 straight-through Gumbel-Softmax。
    strategy_gumbel_tau: float = 1.0
    min_grasp_score: float = 0.3
    selection_mode: str = "soft"  # soft | gumbel | argmax（评测时强制 argmax）

    # 联合损失中辅助项的权重（flow matching 项权重固定为 1.0）。
    loss_weight_contact_temp: float = 0.2
    loss_weight_contact_valid: float = 0.1
    loss_weight_field_temp: float = 0.3
    loss_weight_field_safe: float = 0.1
    loss_weight_skill: float = 0.2
    loss_weight_target: float = 0.3

    def __post_init__(self):
        if self.selection_mode not in ("soft", "gumbel", "argmax"):
            raise ValueError(f"selection_mode 必须是 soft/gumbel/argmax，得到 {self.selection_mode!r}")
        if self.history_size < 1 or self.window_size < 1 or self.signal_dim < 1:
            raise ValueError("history_size / window_size / signal_dim 必须为正整数")
        if self.num_surface_points < 1 or self.num_spatial_tokens < 1:
            raise ValueError("num_surface_points / num_spatial_tokens 必须为正整数")

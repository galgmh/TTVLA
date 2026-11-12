"""Thermal-π0.5 的模块配置。

所有维度（K/L/C/M/D）都从这里读取，模型内部不硬编码；
``enabled=False`` 时 PI0Pytorch 不创建任何热觉模块，行为与原 π0.5 完全一致。
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class ThermalPi0Config:
    """热觉分支的可冻结配置（作为 Pi0Config.thermal 一起序列化）。"""

    # 总开关：False 时原模型行为不变（Stage 0 回归基线）。
    enabled: bool = False

    # 在线输入只有当前滑动窗口[B,L,C]；K只是内部完成接触token的容量。
    memory_size: int = 8
    window_size: int = 25
    signal_dim: int = 14

    # 在线接触事件滞回。一次持续接触只在结束时向MemoryBank写入一次。
    contact_on_threshold: float = 0.7
    contact_off_threshold: float = 0.3
    contact_on_steps: int = 2
    contact_off_steps: int = 2
    active_token_momentum: float = 0.8

    # 新增模块内部维度、热接触 latent 维度、压缩后送入 π0.5 的空间 token 数。
    hidden_dim: int = 256
    thermal_latent_dim: int = 128
    num_spatial_tokens: int = 8

    # 策略辅助头只提供监督与诊断，不再产生进入 prefix 的 goal token。
    num_skills: int = 4  # PROBE / GRASP / RETRACT / ABORT
    field_fourier_bands: int = 6

    # ThermalPrefixAdapter：小门控让新增 token 平滑接入预训练 PaliGemma。
    prefix_adapter_hidden_dim: int = 512
    prefix_gate_init: float = 0.01

    # 联合损失中辅助项的权重（flow matching 项权重固定为 1.0）。
    loss_weight_contact_temp: float = 0.2
    loss_weight_contact_valid: float = 0.1
    loss_weight_field_temp: float = 0.3
    loss_weight_skill: float = 0.2

    def __post_init__(self):
        if self.memory_size < 1 or self.window_size < 1 or self.signal_dim < 1:
            raise ValueError("memory_size / window_size / signal_dim 必须为正整数")
        if not 0 <= self.contact_off_threshold < self.contact_on_threshold <= 1:
            raise ValueError("接触滞回阈值必须满足 0 <= off < on <= 1")
        if self.contact_on_steps < 1 or self.contact_off_steps < 1:
            raise ValueError("contact_on_steps / contact_off_steps 必须为正整数")
        if not 0 <= self.active_token_momentum < 1:
            raise ValueError("active_token_momentum 必须位于 [0,1)")
        if self.num_spatial_tokens < 1 or self.field_fourier_bands < 1:
            raise ValueError("num_spatial_tokens / field_fourier_bands 必须为正整数")
        if self.num_skills != 4:
            raise ValueError("修改版策略固定为 4 类：PROBE/GRASP/RETRACT/ABORT")
        if self.prefix_adapter_hidden_dim < 1:
            raise ValueError("prefix_adapter_hidden_dim 必须为正整数")
        if not 0.0 < self.prefix_gate_init < 1.0:
            raise ValueError("prefix_gate_init 必须位于 (0, 1)")

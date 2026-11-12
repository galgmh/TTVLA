"""独立单触点训练的兼容入口，内部复用 Thermal-π0.5 的唯一接触编码器。"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from models.pi05.thermal.contact_encoder import ThermalContactEncoder
from models.pi05.thermal.outputs import ContactEncoderOutput


class ThermalEncoder(nn.Module):
    """把 ``[B,L,C]`` 适配成共享编码器需要的 ``[B,K=1,L,C]``。

    参数名保留旧训练配置接口，但实现不再维护第二套TCN。四层默认通道为
    ``64,64,128,128``，与 Thermal-π0.5 的默认接触编码器完全一致。
    """

    def __init__(self, *, in_channels: int = 14, hidden: int = 64, levels: int = 4,
                 kernel: int = 3, dropout: float = 0.0, out_dim: int = 128):
        super().__init__()
        if levels < 1:
            raise ValueError("levels 必须 >= 1")
        del dropout  # 共享编码器当前不使用dropout；保留参数仅兼容旧YAML。
        hidden_channels = tuple(hidden if i < 2 else 2 * hidden for i in range(levels))
        dilations = tuple(2**i for i in range(levels))
        self.core = ThermalContactEncoder(
            signal_dim=in_channels,
            hidden_channels=hidden_channels,
            dilations=dilations,
            kernel_size=kernel,
            latent_dim=out_dim,
        )
        self.in_channels = int(in_channels)
        self.out_dim = int(out_dim)
        self.kernel = int(kernel)
        self.dilations = dilations

    @property
    def receptive_field(self) -> int:
        # 每个 residual block 有两层同dilation卷积。
        return 1 + 2 * (self.kernel - 1) * sum(self.dilations)

    def forward_output(self, features: torch.Tensor, mask: torch.Tensor) -> ContactEncoderOutput:
        if features.dim() != 3:
            raise ValueError(f"features 应为 (B,L,C)，实为 {tuple(features.shape)}")
        if features.shape[-1] != self.in_channels:
            raise ValueError(f"特征通道 {features.shape[-1]} 与配置 {self.in_channels} 不一致")
        if mask.shape != features.shape[:2]:
            raise ValueError(f"mask 应为 {tuple(features.shape[:2])}，实为 {tuple(mask.shape)}")
        return self.core(features, mask.bool())

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.forward_output(features, mask).latent


def load_contact_encoder_checkpoint(
    target: ThermalContactEncoder,
    checkpoint: Mapping[str, object],
) -> None:
    """把独立预训练checkpoint严格加载到VLA的``contact_encoder``。

    接受训练脚本保存的完整字典或直接state_dict；只转换固定前缀
    ``encoder.core.``，任何缺失/多余键都会报错。
    """
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint['model'] 必须是 state_dict 映射")
    prefix = "encoder.core."
    converted = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    if not converted:
        raise ValueError("checkpoint 中没有 encoder.core.* 权重")
    target.load_state_dict(converted, strict=True)


__all__ = ["ThermalEncoder", "load_contact_encoder_checkpoint"]

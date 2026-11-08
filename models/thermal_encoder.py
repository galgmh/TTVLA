"""因果膨胀卷积编码器：逐帧 14 维序列 -> z ∈ R^out_dim。

* 卷积核向左补零（只看当前与历史），感受野 ``1 + (kernel-1)·Σdilation``；
  4 层膨胀率 1/2/4/8、kernel 3 时是 31 帧，50 Hz 下 0.62 s。
* mask 之外的帧在每一层之后都清零：前缀窗口的前端是补零，偏置项会让它
  变成非零，必须显式清掉，否则“历史长度”会通过偏置泄漏进特征。
* 输出取每个样本*最后一个有效帧*的隐状态，而不是固定末帧。
"""

from __future__ import annotations

import torch
from torch import nn


class CausalBlock(nn.Module):
    """一层因果卷积 + 逐位置归一化 + 残差。

    归一化用 ``LayerNorm`` 在通道维做：每个时刻只统计自己的通道，
    被 mask 掉的位置（清零后）不会把统计量带偏，也不会随前缀长度漂移。
    """

    def __init__(self, channels: int, *, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel, dilation=dilation)
        self.norm = nn.LayerNorm(channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(nn.functional.pad(x, (self.pad, 0)))
        # Conv1d 是 (B,C,H)，LayerNorm 要按最后一维（通道）统计，先转置。
        y = self.norm(y.transpose(1, 2)).transpose(1, 2)
        return self.drop(self.act(y)) + x


class ThermalEncoder(nn.Module):
    """序列编码器（因果、膨胀、残差）。"""

    def __init__(
        self,
        *,
        in_channels: int = 14,
        hidden: int = 64,
        levels: int = 4,
        kernel: int = 3,
        dropout: float = 0.1,
        out_dim: int = 128,
    ):
        super().__init__()
        if levels < 1:
            raise ValueError("levels 必须 >= 1")
        self.in_channels = int(in_channels)
        self.input_proj = nn.Conv1d(in_channels, hidden, 1)
        self.blocks = nn.ModuleList(
            CausalBlock(hidden, kernel=kernel, dilation=2**i, dropout=dropout) for i in range(levels)
        )
        self.output = nn.Linear(hidden, out_dim)
        self.out_dim = int(out_dim)

    @property
    def receptive_field(self) -> int:
        return 1 + sum(block.pad for block in self.blocks)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``features`` (B,H,C)，``mask`` (B,H) -> ``z`` (B,out_dim)。"""
        if features.dim() != 3:
            raise ValueError(f"features 应为 (B,H,C)，实为 {tuple(features.shape)}")
        if features.shape[-1] != self.in_channels:
            raise ValueError(f"特征通道 {features.shape[-1]} 与配置 {self.in_channels} 不一致")
        keep = mask.unsqueeze(1)
        x = self.input_proj(features.transpose(1, 2)) * keep
        for block in self.blocks:
            x = block(x) * keep
        # 窗口是右对齐的（历史不足时在前端补零），所以最后一个有效帧就是
        # 第 H-1 帧；按“最后一个 True”取，避免窗口组织方式变化时悄悄取错。
        flipped = mask.flip(1).to(torch.int64)
        last_idx = (mask.size(1) - 1 - torch.argmax(flipped, dim=1)).view(-1, 1, 1)
        last = x.gather(2, last_idx.expand(-1, x.size(1), 1)).squeeze(-1)
        return self.output(last)


__all__ = ["CausalBlock", "ThermalEncoder"]

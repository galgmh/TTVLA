"""因果时序卷积：把每次短时接触的温度/力序列编码成局部热觉 token。

输入只包含传感器时序（[B, K, L, C]）与时间 mask；接触坐标、候选点和
link 身份都不进入该编码器，防止用位置代替热响应完成温度预测。
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

from models.pi05.thermal.outputs import ContactEncoderOutput


class _CausalResidualBlock(nn.Module):
    """因果残差卷积块：时刻 t 的输出只依赖 <= t 的输入。

    ``nn.Conv1d(padding=d)`` 会在两侧各补 d，导致看到未来帧；这里改为
    只在左侧补 ``(kernel_size - 1) * dilation``。
    """

    def __init__(self, in_channels: int, out_channels: int, dilation: int, kernel_size: int = 3):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        self.act = nn.GELU()
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [M, C, T]；mask: [M, 1, T]（float，无效帧为 0）
        h = self.act(self.conv1(F.pad(x, (self.left_pad, 0)))) * mask
        h = self.conv2(F.pad(h, (self.left_pad, 0)))
        return (self.skip(x) + h) * mask


class ThermalContactEncoder(nn.Module):
    """4 层因果膨胀卷积 + masked mean/last-valid pooling。

    输入:
        history:   [B, K, L, C]
        time_mask: [B, K, L]，True 表示该帧有效
        contact_mask: [B, K]，True 表示该次历史接触存在（K 允许为 0）
    """

    def __init__(
        self,
        signal_dim: int = 14,
        hidden_channels: tuple[int, ...] = (64, 64, 128, 128),
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        kernel_size: int = 3,
        latent_dim: int = 128,
    ):
        super().__init__()
        if len(hidden_channels) != len(dilations):
            raise ValueError("hidden_channels 与 dilations 长度必须一致")
        blocks = []
        in_channels = signal_dim
        for out_channels, dilation in zip(hidden_channels, dilations, strict=True):
            blocks.append(_CausalResidualBlock(in_channels, out_channels, dilation, kernel_size))
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        last = hidden_channels[-1]
        # mean 与 last-valid 两个池化结果拼接后再投影到 latent。
        self.project = nn.Sequential(
            nn.Linear(2 * last, last),
            nn.GELU(),
            nn.Linear(last, latent_dim),
        )
        self.temp_mean_head = nn.Linear(latent_dim, 1)
        self.temp_logvar_head = nn.Linear(latent_dim, 1)
        self.valid_head = nn.Linear(latent_dim, 1)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [M, C, T]；mask: [M, T] bool
        w = mask.to(x.dtype).unsqueeze(1)
        return (x * w).sum(dim=-1) / w.sum(dim=-1).clamp_min(1.0)

    @staticmethod
    def _last_valid(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # 注意窗口可能右对齐（有效帧在尾部），也可能整体无效，这里按真实 mask 取最后一个 True。
        m = mask.bool()
        positions = torch.arange(m.shape[-1], device=m.device).expand_as(m)
        last = torch.where(m, positions, torch.full_like(positions, -1)).max(dim=-1).values  # [M]
        gather = last.clamp_min(0).view(-1, 1, 1).expand(-1, x.shape[1], 1)
        pooled = x.gather(-1, gather).squeeze(-1)
        return pooled * (last >= 0).to(x.dtype).unsqueeze(-1)

    def forward(
        self,
        history: torch.Tensor,
        time_mask: torch.Tensor,
        contact_mask: torch.Tensor,
    ) -> ContactEncoderOutput:
        b, k, length, channels = history.shape
        if k == 0:
            # K=0 是合法输入（尚无历史接触）：直接返回空张量，后续由 NO_CONTACT token 处理。
            zeros = history.new_zeros((b, 0, self.temp_mean_head.in_features))
            empty = history.new_zeros((b, 0, 1))
            return ContactEncoderOutput(
                latent=zeros,
                temp_mean_c=empty,
                temp_logvar=empty.clone(),
                valid_logit=empty.clone(),
            )

        h = history.reshape(b * k, length, channels).transpose(1, 2)  # [M, C, L]
        # 无效接触整体屏蔽：不影响其它接触，也不产生虚假池化。
        valid_contact = contact_mask.reshape(b * k, 1, 1).to(h.dtype)
        m = time_mask.reshape(b * k, length).to(h.dtype).unsqueeze(1) * valid_contact
        for block in self.blocks:
            h = block(h, m)

        mask_bool = m.squeeze(1) > 0.5
        pooled = torch.cat([self._masked_mean(h, mask_bool), self._last_valid(h, mask_bool)], dim=-1)

        latent = self.project(pooled).view(b, k, -1)
        return ContactEncoderOutput(
            latent=latent,
            temp_mean_c=self.temp_mean_head(latent),
            temp_logvar=self.temp_logvar_head(latent),
            valid_logit=self.valid_head(latent),
        )

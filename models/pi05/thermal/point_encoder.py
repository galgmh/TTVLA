"""候选表面点编码器：归一化坐标 + Fourier 特征 + 可达/抓取几何特征。"""

from __future__ import annotations

import torch
from torch import nn


class SurfacePointEncoder(nn.Module):
    """把每个候选点编码成 256 维表面 token。

    输入（均在物体坐标系，xyz 已归一化）:
        xyz:          [B, N, 3]
        normal:       [B, N, 3]
        reachable:    [B, N]    bool
        grasp_score:  [B, N, 1]
        ee_xyz:       [B, 3]    末端执行器位置，用于距离特征
    """

    def __init__(self, hidden_dim: int = 256, bands: int = 6):
        super().__init__()
        self.bands = bands
        # 3(xyz) + 3*2*bands(Fourier) + 3(normal) + 1(reachable) + 1(grasp) + 1(distance)
        in_dim = 3 * (1 + 2 * bands) + 3 + 1 + 1 + 1
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        xyz: torch.Tensor,
        normal: torch.Tensor,
        reachable: torch.Tensor,
        grasp_score: torch.Tensor,
        ee_xyz: torch.Tensor,
    ) -> torch.Tensor:
        device = xyz.device
        freqs = (2.0 ** torch.arange(self.bands, device=device, dtype=xyz.dtype)) * torch.pi
        phase = xyz.unsqueeze(-1) * freqs  # [B, N, 3, bands]
        fourier = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1).flatten(2)  # [B, N, 6*bands]
        distance = torch.linalg.norm(xyz - ee_xyz.unsqueeze(1), dim=-1, keepdim=True)  # [B, N, 1]
        features = torch.cat(
            [xyz, fourier, normal, reachable.to(xyz.dtype).unsqueeze(-1), grasp_score.to(xyz.dtype), distance],
            dim=-1,
        )
        return self.mlp(features)

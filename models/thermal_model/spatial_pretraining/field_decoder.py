"""训练期空间热场辅助解码器。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def _fourier(xyz: torch.Tensor, bands: int) -> torch.Tensor:
    freqs = (2.0 ** torch.arange(bands, device=xyz.device, dtype=xyz.dtype)) * torch.pi
    phase = xyz.unsqueeze(-1) * freqs
    return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1).flatten(-2)


@dataclass
class FieldDecoderOutput:
    temp_mean_c: torch.Tensor
    temp_logvar: torch.Tensor


class SpatialFieldDecoder(nn.Module):
    """从空间记忆中读取任意查询点的温度分布。"""

    def __init__(
        self,
        hidden_dim: int = 256,
        bands: int = 6,
        num_heads: int = 8,
        logvar_min: float = -4.0,
        logvar_max: float = 3.0,
    ):
        super().__init__()
        if not logvar_min < logvar_max:
            raise ValueError("logvar_min 必须小于 logvar_max")
        self.bands = bands
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.query_proj = nn.Sequential(
            nn.Linear(3 + 6 * bands, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.temp_mean_head = nn.Linear(hidden_dim, 1)
        self.temp_logvar_head = nn.Linear(hidden_dim, 1)
        nn.init.constant_(self.temp_logvar_head.bias, 2.0)

    def forward(
        self,
        spatial_latents: torch.Tensor,
        query_xyz_base: torch.Tensor,
    ) -> FieldDecoderOutput:
        query = self.query_proj(
            torch.cat([query_xyz_base, _fourier(query_xyz_base, self.bands)], dim=-1)
        )
        memory = self.memory_norm(spatial_latents)
        attended, _ = self.cross_attn(
            self.query_norm(query),
            memory,
            memory,
            need_weights=False,
        )
        features = self.output_norm(query + attended)
        return FieldDecoderOutput(
            temp_mean_c=self.temp_mean_head(features),
            temp_logvar=self.logvar_min
            + (self.logvar_max - self.logvar_min)
            * torch.sigmoid(self.temp_logvar_head(features)),
        )


__all__ = ["FieldDecoderOutput", "SpatialFieldDecoder"]

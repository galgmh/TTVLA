"""训练/评测用连续热场查询解码器；不参与运行时触点选择。"""

from __future__ import annotations

import torch
from torch import nn

from models.pi05.thermal.outputs import FieldDecoderOutput


def _fourier(xyz: torch.Tensor, bands: int) -> torch.Tensor:
    freqs = (2.0 ** torch.arange(bands, device=xyz.device, dtype=xyz.dtype)) * torch.pi
    phase = xyz.unsqueeze(-1) * freqs
    return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1).flatten(-2)


class SpatialFieldDecoder(nn.Module):
    """用任意训练查询坐标读取contact memory并预测温度mean/logvar。"""

    def __init__(self, hidden_dim: int = 256, bands: int = 6, num_heads: int = 8):
        super().__init__()
        self.bands = bands
        self.query_proj = nn.Sequential(
            nn.Linear(3 + 6 * bands, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.temp_mean_head = nn.Linear(hidden_dim, 1)
        self.temp_logvar_head = nn.Linear(hidden_dim, 1)

    def forward(self, spatial_latents: torch.Tensor, query_xyz_base: torch.Tensor) -> FieldDecoderOutput:
        query = self.query_proj(torch.cat([query_xyz_base, _fourier(query_xyz_base, self.bands)], dim=-1))
        attended, _ = self.cross_attn(
            self.query_norm(query), self.memory_norm(spatial_latents), self.memory_norm(spatial_latents),
            need_weights=False,
        )
        features = self.output_norm(query + attended)
        return FieldDecoderOutput(
            temp_mean_c=self.temp_mean_head(features),
            temp_logvar=self.temp_logvar_head(features),
        )

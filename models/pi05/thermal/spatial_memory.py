"""热觉空间记忆：只压缩已完成接触，不接收未来候选表面点。"""

from __future__ import annotations

import torch
from torch import nn

from models.pi05.thermal.outputs import SpatialMemoryOutput


def _fourier(xyz: torch.Tensor, bands: int) -> torch.Tensor:
    """6 频带 Fourier 特征：[B, ..., 3] -> [B, ..., 6*bands]。"""
    freqs = (2.0 ** torch.arange(bands, device=xyz.device, dtype=xyz.dtype)) * torch.pi
    phase = xyz.unsqueeze(-1) * freqs
    return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1).flatten(-2)


class ContactTokenBuilder(nn.Module):
    """把一次接触的 latent 与几何/温度/时效特征拼成 256 维 contact token。

    温度均值/标准差来自 ThermalContactEncoder 的预测头，而不是特权标签。
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        bands: int = 6,
    ):
        super().__init__()
        self.bands = bands
        # latent + Fourier(xyz) + normal + temp_mean + temp_std + valid_prob
        in_dim = latent_dim + 6 * bands + 3 + 1 + 1 + 1
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        latent: torch.Tensor,
        xyz: torch.Tensor,
        normal: torch.Tensor,
        temp_mean_c: torch.Tensor,
        temp_logvar: torch.Tensor,
        valid_prob: torch.Tensor,
    ) -> torch.Tensor:
        temp_std = torch.exp(0.5 * temp_logvar.clamp(-10.0, 10.0))
        features = torch.cat(
            [
                latent,
                _fourier(xyz, self.bands),
                normal,
                temp_mean_c,
                temp_std,
                valid_prob,
            ],
            dim=-1,
        )
        return self.mlp(features)


class _TransformerLayer(nn.Module):
    """pre-norm 残差注意力 + FFN；self-attention 与 cross-attention 共用。"""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0, mlp_ratio: int = 4):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * dim, dim),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        # key_padding_mask: True 表示忽略该 key
        out, _ = self.attn(
            self.norm_q(q),
            self.norm_kv(kv),
            self.norm_kv(kv),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        h = q + out
        return h + self.ff(self.norm_ff(h))


class ThermalSpatialMemory(nn.Module):
    """Set Transformer融合历史接触，再以M个learned query压缩为prefix memory。"""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_spatial_tokens: int = 8,
        num_heads: int = 8,
        dropout: float = 0.0,
        num_layers: int = 2,
        num_perceiver_layers: int = 2,
    ):
        super().__init__()
        self.no_contact = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.contact_layers = nn.ModuleList(
            [_TransformerLayer(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.perceiver_queries = nn.Parameter(torch.randn(1, num_spatial_tokens, hidden_dim) * 0.02)
        self.perceiver_layers = nn.ModuleList(
            [_TransformerLayer(hidden_dim, num_heads, dropout) for _ in range(num_perceiver_layers)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        contact_tokens: torch.Tensor,
        contact_mask: torch.Tensor,
    ) -> SpatialMemoryOutput:
        # contact_tokens: [B, K, D]；contact_mask: [B, K] bool
        b = contact_tokens.shape[0]
        batch_ones = torch.ones(b, 1, dtype=torch.bool, device=contact_tokens.device)

        tokens = torch.cat([self.no_contact.expand(b, -1, -1), contact_tokens], dim=1)
        contact_valid = torch.cat([batch_ones, contact_mask.bool()], dim=1)
        for layer in self.contact_layers:
            tokens = layer(tokens, tokens, ~contact_valid)

        # learned queries只查询已发生接触；NO_CONTACT始终有效，空历史也不会NaN。
        queries = self.perceiver_queries.expand(b, -1, -1)
        for layer in self.perceiver_layers:
            queries = layer(queries, tokens, ~contact_valid)
        spatial_latents = self.output_norm(queries)

        return SpatialMemoryOutput(spatial_latents=spatial_latents)

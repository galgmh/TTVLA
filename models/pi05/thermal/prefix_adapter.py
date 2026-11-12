"""把热空间记忆平滑映射到 PaliGemma prefix token 空间。"""

from __future__ import annotations

import math

import torch
from torch import nn


class ThermalPrefixAdapter(nn.Module):
    """LayerNorm + MLP + type embedding + 可学习小门控。

    门控只用于 thermal-enabled 模型的稳定接入；严格 π0.5 基线通过完全不创建、
    不追加 thermal token 实现，而不是依赖 gate=0。
    """

    def __init__(
        self,
        hidden_dim: int,
        paligemma_width: int,
        adapter_hidden_dim: int = 512,
        gate_init: float = 0.01,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, adapter_hidden_dim),
            nn.GELU(),
            nn.Linear(adapter_hidden_dim, paligemma_width),
        )
        self.type_embedding = nn.Parameter(torch.zeros(1, 1, paligemma_width))
        gate_logit = math.log(gate_init / (1.0 - gate_init))
        self.gate_logit = nn.Parameter(torch.tensor(gate_logit, dtype=torch.float32))

    @property
    def gate(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit)

    def forward(self, spatial_latents: torch.Tensor) -> torch.Tensor:
        tokens = self.mlp(self.norm(spatial_latents)) + self.type_embedding
        return tokens * self.gate.to(dtype=tokens.dtype, device=tokens.device)

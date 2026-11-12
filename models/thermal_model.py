"""共享接触编码器的独立预训练包装：latent + 温度分布 + 有效性。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from models.thermal_encoder import ThermalEncoder


@dataclass
class ThermalOutput:
    z: torch.Tensor
    temperature: torch.Tensor
    temp_logvar: torch.Tensor
    valid_logit: torch.Tensor


class ThermalModel(nn.Module):
    """6000集单触点预训练与VLA Stage 1共用同一编码器和预测头。"""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        model = dict(config["model"])
        self.encoder = ThermalEncoder(
            in_channels=int(model["in_channels"]),
            hidden=int(model["hidden"]),
            levels=int(model["levels"]),
            kernel=int(model["kernel"]),
            dropout=float(model["dropout"]),
            out_dim=int(model["out_dim"]),
        )

    @property
    def out_dim(self) -> int:
        return self.encoder.out_dim

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> ThermalOutput:
        output = self.encoder.forward_output(features, mask)
        return ThermalOutput(
            z=output.latent,
            temperature=output.temp_mean_c.squeeze(-1),
            temp_logvar=output.temp_logvar.squeeze(-1),
            valid_logit=output.valid_logit.squeeze(-1),
        )


__all__ = ["ThermalModel", "ThermalOutput"]

"""热模型：编码器 + 两个头，一个 ``forward`` 出全部输出。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from models.thermal_encoder import ThermalEncoder
from models.thermal_heads import ContactValidityHead, TemperatureHead


@dataclass
class ThermalOutput:
    """一次前向的三个输出。"""

    z: torch.Tensor  # (B, out_dim) 编码
    temperature: torch.Tensor  # (B,) 预测温度 °C
    valid_logit: torch.Tensor  # (B,) 本帧杯子是否有接触的 logit


class ThermalModel(nn.Module):
    """序列 -> z -> 温度 / 接触有效性。"""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        model = dict(config["model"])
        hidden = int(model["hidden"])
        self.encoder = ThermalEncoder(
            in_channels=int(model["in_channels"]),
            hidden=hidden,
            levels=int(model["levels"]),
            kernel=int(model["kernel"]),
            dropout=float(model["dropout"]),
            out_dim=int(model["out_dim"]),
        )
        self.temperature_head = TemperatureHead(self.encoder.out_dim, hidden)
        self.validity_head = ContactValidityHead(self.encoder.out_dim, hidden)

    @property
    def out_dim(self) -> int:
        return self.encoder.out_dim

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> ThermalOutput:
        z = self.encoder(features, mask)
        return ThermalOutput(
            z=z,
            temperature=self.temperature_head(z),
            valid_logit=self.validity_head(z),
        )


__all__ = ["ThermalModel", "ThermalOutput"]

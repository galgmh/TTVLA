"""热编码器的两个输出头。

* 温度头     ``T̂ = f_T(z)``           —— 被接触 link 的当前温度，Huber 监督；
* 有效性头   ``ĉ = sigmoid(f_c(z))``   —— 本帧杯子是否有接触，BCE。
"""

from __future__ import annotations

import torch
from torch import nn


class MLPHead(nn.Module):
    """``in_dim -> hidden -> 1`` 的两层头。"""

    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TemperatureHead(MLPHead):
    """温度回归头：输出 °C。"""


class ContactValidityHead(MLPHead):
    """接触有效性头：输出 logit。"""


__all__ = ["ContactValidityHead", "MLPHead", "TemperatureHead"]

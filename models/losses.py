"""损失：温度 Huber + 接触有效性 BCE。

总损失 ``L = L_temp + 0.2·L_valid``（权重来自模型配置）。

标签里的 NaN 是“该样本没有监督信号”（混杂接触集没有单一接触 link、
无接触帧没有温度标签），计算时按 ``isfinite`` 过滤，而不是把 NaN 当 0。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LossTerms:
    """一次前向的两项损失与样本数（样本数为 0 时对应项为 0）。"""

    temp: torch.Tensor
    valid: torch.Tensor
    temp_count: int
    valid_count: int

    def total(self, weights: dict[str, float]) -> torch.Tensor:
        return float(weights["temp"]) * self.temp + float(weights["valid"]) * self.valid


def masked_huber(pred: torch.Tensor, target: torch.Tensor, *, delta: float = 1.0) -> tuple[torch.Tensor, int]:
    """只在 ``target`` 有限的样本上算 Huber；全无效时返回 0。"""
    keep = torch.isfinite(target)
    count = int(keep.sum().item())
    if count == 0:
        return pred.sum() * 0.0, 0
    return F.huber_loss(pred[keep], target[keep], delta=delta), count


def masked_bce(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, int]:
    """只在 ``target`` 有限的样本上算 BCE-with-logits；全无效时返回 0。"""
    keep = torch.isfinite(target)
    count = int(keep.sum().item())
    if count == 0:
        return logits.sum() * 0.0, 0
    return F.binary_cross_entropy_with_logits(logits[keep], target[keep]), count


def compute_losses(output, batch, *, huber_delta: float = 1.0) -> LossTerms:
    """``output`` 是 ``ThermalOutput``，``batch`` 是张量字典。"""
    temp, temp_count = masked_huber(output.temperature, batch["temperature"], delta=huber_delta)
    valid, valid_count = masked_bce(output.valid_logit, batch["contact_valid"].to(output.valid_logit.dtype))
    return LossTerms(
        temp=temp,
        valid=valid,
        temp_count=temp_count,
        valid_count=valid_count,
    )


__all__ = ["LossTerms", "compute_losses", "masked_bce", "masked_huber"]

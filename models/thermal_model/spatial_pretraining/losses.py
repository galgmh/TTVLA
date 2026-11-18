"""空间热觉表示的查询点温度损失。"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LossTerms:
    field_temp: torch.Tensor
    field_count: int

    def total(self, weights: dict[str, float]) -> torch.Tensor:
        return float(weights["field_temp"]) * self.field_temp


def masked_gaussian_nll(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """只在有效查询点上计算 Gaussian NLL。"""
    keep = mask.bool() & torch.isfinite(target)
    count = int(keep.sum().item())
    if count == 0:
        return (mean.sum() + logvar.sum()) * 0.0, 0
    lv = logvar[keep]
    variance = torch.exp(lv) + 1e-6
    nll = 0.5 * (lv + (target[keep] - mean[keep]).square() / variance)
    return nll.mean(), count


def compute_losses(output, batch) -> LossTerms:
    target = batch["query_temperature"].to(
        device=output.field.temp_mean_c.device,
        dtype=output.field.temp_mean_c.dtype,
    )
    mask = batch["query_mask"].to(device=target.device).bool()
    field_temp, field_count = masked_gaussian_nll(
        output.field.temp_mean_c,
        output.field.temp_logvar,
        target,
        mask,
    )
    return LossTerms(field_temp=field_temp, field_count=field_count)


__all__ = ["LossTerms", "compute_losses", "masked_gaussian_nll"]

"""Thermal-π0.5 辅助损失。

所有项都按 mask 归一化，并返回逐样本标量 [B]，方便：
1. ``compute_loss_dict`` 汇总成训练日志标量；
2. 原 ``forward`` 把辅助项广播加回 flow loss 张量。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812

from models.pi05.thermal.outputs import IntegratedThermalOutput
from models.pi05.thermal.outputs import ThermalSupervision


def _masked_mean_per_sample(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """values/mask: [B, ...] -> [B]；全 mask 的样本返回 0。"""
    w = mask.to(values.dtype).reshape(values.shape[0], -1)
    v = values.reshape(values.shape[0], -1)
    return (v * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)


def masked_gaussian_nll(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """均值/对数方差预测的高斯负对数似然（忽略常数项），返回 [B]。"""
    logvar = logvar.clamp(-10.0, 10.0)
    variance = torch.exp(logvar) + 1e-6
    nll = 0.5 * (logvar + (target - mean) ** 2 / variance)
    return _masked_mean_per_sample(nll.squeeze(-1), mask)


def masked_bce_with_logits(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1), target.squeeze(-1), reduction="none")
    return _masked_mean_per_sample(loss, mask)


def compute_thermal_losses(
    output: IntegratedThermalOutput,
    supervision: ThermalSupervision,
    weights: dict[str, float],
) -> dict[str, torch.Tensor]:
    """计算接触温度/有效性、查询热场和全局技能损失。"""
    device = output.contact.temp_mean_c.device
    contact_target = supervision.contact_temp_c.to(device)
    contact_valid = supervision.contact_valid.to(device)
    contact_mask = contact_valid.squeeze(-1) > 0.5
    field_mask = supervision.field_query_mask.to(device).bool()
    strategy_type = supervision.strategy_type.to(device).long()

    if output.field is None:
        raise ValueError("计算field_temp损失前必须用field_query_xyz_base运行SpatialFieldDecoder")

    terms = {
        "contact_temp": masked_gaussian_nll(
            output.contact.temp_mean_c,
            output.contact.temp_logvar,
            contact_target,
            contact_mask,
        ),
        "contact_valid": masked_bce_with_logits(
            output.contact.valid_logit,
            contact_valid,
            torch.ones_like(contact_mask),
        ),
        "field_temp": masked_gaussian_nll(
            output.field.temp_mean_c,
            output.field.temp_logvar,
            supervision.field_query_temp_c.to(device),
            field_mask,
        ),
    }
    if output.strategy is None:
        zero = torch.zeros_like(terms["contact_temp"])
        terms["skill"] = zero
    else:
        terms["skill"] = F.cross_entropy(output.strategy.skill_logits, strategy_type, reduction="none")
    names = ("contact_temp", "contact_valid", "field_temp", "skill")
    aux_total = torch.zeros_like(terms["skill"])
    for name in names:
        aux_total = aux_total + weights[name] * terms[name]
    terms["aux_total"] = aux_total
    return terms

"""Auxiliary losses used while training a thermal-enabled π0.5 policy."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812

from thermal_model.pi05_training.labels import ThermalSupervision


def _masked_mean_per_sample(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(values.dtype).reshape(values.shape[0], -1)
    flat = values.reshape(values.shape[0], -1)
    return (flat * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)


def masked_gaussian_nll(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    nll = 0.5 * (logvar + (target - mean).square() / (torch.exp(logvar) + 1e-6))
    return _masked_mean_per_sample(nll.squeeze(-1), mask)


def masked_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(
        logits.squeeze(-1),
        target.squeeze(-1),
        reduction="none",
    )
    return _masked_mean_per_sample(loss, mask)


def compute_thermal_losses(
    output,
    supervision: ThermalSupervision,
    weights: dict[str, float],
    *,
    strategy=None,
) -> dict[str, torch.Tensor]:
    """计算最终模型的接触监督；strategy 由训练侧显式传入。"""
    device = output.contact.temp_mean_c.device
    contact_valid = supervision.contact_valid.to(device)
    contact_mask = contact_valid.squeeze(-1) > 0.5
    terms = {
        "contact_temp": masked_gaussian_nll(
            output.contact.temp_mean_c,
            output.contact.temp_logvar,
            supervision.contact_temp_c.to(device),
            contact_mask,
        ),
        "contact_valid": masked_bce_with_logits(
            output.contact.valid_logit,
            contact_valid,
            torch.ones_like(contact_mask),
        ),
    }
    if strategy is not None:
        terms["skill"] = F.cross_entropy(
            strategy.skill_logits,
            supervision.strategy_type.to(device).long(),
            reduction="none",
        )
    terms["aux_total"] = sum(
        float(weights.get(name, 1.0)) * value
        for name, value in terms.items()
        if name != "aux_total"
    )
    return terms


__all__ = [
    "compute_thermal_losses",
    "masked_bce_with_logits",
    "masked_gaussian_nll",
]

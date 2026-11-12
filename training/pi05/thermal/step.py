"""Thermal-π0.5 训练 wrapper：统一走 ``compute_loss_dict`` 并分离日志量。

后续 Genesis 数据管线（Task G）产出的 batch 只要带上 ``ThermalSupervision``
即可直接调用本函数，无需修改训练脚本的损失逻辑。
"""

from __future__ import annotations

import torch

from models.pi05.thermal.outputs import ThermalSupervision


def thermal_train_step(
    model,
    observation,
    actions: torch.Tensor,
    supervision: ThermalSupervision | None = None,
    *,
    noise: torch.Tensor | None = None,
    time: torch.Tensor | None = None,
    loss_keys: tuple[str, ...] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """返回 ``(total_loss, metrics)``：total 保留计算图，metrics 全部 detach。

    ``supervision=None`` 时只计算 flow loss（例如 LIBERO 这种没有热觉标签的
    数据）；此时 ``loss_keys`` 只能是 ``("flow",)``。
    ``loss_keys`` 给定时只把这些分项相加作为反传目标（例如阶段 1 只回传
    ``contact_temp/contact_valid/field_temp``）；metrics 仍包含全部
    分项，方便日志对比。None 表示使用 ``compute_loss_dict`` 的 ``total``。
    """
    physical_keys = {"contact_temp", "contact_valid", "field_temp"}
    thermal_only = loss_keys is not None and set(loss_keys) <= physical_keys
    if thermal_only:
        losses = model.compute_thermal_only_loss_dict(observation, supervision)
    else:
        losses = model.compute_loss_dict(observation, actions, supervision, noise=noise, time=time)
    metrics = {name: value.detach() for name, value in losses.items()}
    if loss_keys is None:
        return losses["total"], metrics
    if supervision is None and any(name != "flow" for name in loss_keys):
        raise ValueError("缺少 ThermalSupervision，只能使用 flow loss")
    unknown = [name for name in loss_keys if name not in losses]
    if unknown:
        raise ValueError(f"未知损失项 {unknown}；可选 {sorted(losses)}")
    # flow 本身权重为 1；thermal辅助项使用配置权重，避免阶段训练绕过YAML权重。
    objective_terms = [losses[name] if name == "flow" else losses[f"weighted_{name}"] for name in loss_keys]
    total = sum(objective_terms)
    return total, metrics

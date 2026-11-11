"""Thermal-π0.5 评测入口（纯 PyTorch）：在数据集上计算 flow + 热觉分项损失。

用法::

    python -m training.pi05.eval --config configs/training/pi05_thermal_stage3.yaml \
        [--checkpoint checkpoints/pi05_thermal/stage3] [--data-root DATASET] \
        [--num-batches 20] [--device cpu]

这里做的是**离线损失评测**（与训练同一套前向、关闭梯度）。闭环成功率评测
需要仿真侧的 ``thermalvla-1.0`` 导出器与环境适配器，尚未实现。
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import pathlib

import torch

from models.pi05.observation import Observation
from models.pi05.pi0_pytorch import PI0Pytorch
from training.pi05.thermal.labels import build_thermal_supervision
from training.pi05.checkpoints import load_checkpoint
from training.pi05.config import TrainConfig, load_train_config
from training.pi05.data import build_dataloader
from training.pi05.train import _to_device, init_logging, resolve_device, set_seed

logger = logging.getLogger("thermal_pi05.eval")


def evaluate(
    config: TrainConfig,
    *,
    checkpoint_dir: str | pathlib.Path | None = None,
    data_root: str | pathlib.Path | None = None,
    num_batches: int = 20,
    device: str = "auto",
) -> dict[str, float]:
    """加载检查点，在 ``data_root`` 上计算平均损失分项。

    返回 ``{分项: 平均值, "batches": 批次数, "step": 检查点步数}``。
    """
    if num_batches < 1:
        raise ValueError(f"num_batches 必须为正整数，得到 {num_batches}")
    if data_root is not None:
        config = dataclasses.replace(config, data=dataclasses.replace(config.data, root=str(data_root)))
    torch_device = resolve_device(device)
    set_seed(config.seed)
    loader, _ = build_dataloader(config, shuffle=False)
    model = PI0Pytorch(config.model).to(torch_device)
    step = load_checkpoint(
        model=model,
        optimizer=None,
        checkpoint_dir=checkpoint_dir or config.checkpoint_path,
        device=torch_device,
    )
    model.eval()
    logger.info(
        "已加载检查点 step=%d；在 %s 上评测 %d 个 batch",
        step,
        config.data.root or config.data.kind,
        num_batches,
    )

    totals: dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for batch in loader:
            if count >= num_batches:
                break
            batch = _to_device(batch, torch_device)
            supervision_arrays = batch.pop("thermal_supervision", None)
            actions = batch.pop("actions")
            observation = Observation.from_dict(batch)
            supervision = (
                build_thermal_supervision(supervision_arrays, device=torch_device)
                if supervision_arrays is not None
                else None
            )
            losses = model.compute_loss_dict(observation, actions, supervision)
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value)
            count += 1

    if count == 0:
        raise RuntimeError("评测数据为空，无法计算损失")
    averages = {name: value / count for name, value in totals.items()}
    averages["batches"] = float(count)
    averages["step"] = float(step)
    return averages


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在数据上评估 Thermal-π0.5 的损失分项")
    parser.add_argument("--config", required=True, help="与训练一致的 YAML 配置")
    parser.add_argument("--checkpoint", default=None, help="检查点目录（默认用配置里的 checkpoint_dir）")
    parser.add_argument("--data-root", default=None, help="覆盖 data.root（评估集目录）")
    parser.add_argument("--num-batches", type=int, default=20, help="最多评估多少个 batch")
    parser.add_argument("--device", default=None, help="cpu / cuda / auto（默认用配置里的 device）")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    init_logging()
    args = parse_args(argv)
    config = load_train_config(args.config)
    metrics = evaluate(
        config,
        checkpoint_dir=args.checkpoint,
        data_root=args.data_root,
        num_batches=args.num_batches,
        device=args.device or config.device,
    )
    parts = []
    for name, value in sorted(metrics.items()):
        if name in ("batches", "step"):
            parts.append(f"{name}={int(value)}")
        else:
            parts.append(f"{name}={value:.4f}")
    logger.info("评测结果：%s", "  ".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["evaluate", "main", "parse_args"]

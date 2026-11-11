"""Thermal-π0.5 训练入口（参考 openpi `scripts/train_pytorch.py` 重写）。

单卡训练/微调，语义与官方 PyTorch 训练器一致：

* warmup + cosine 学习率（每个参数组按 ``lr_scale`` 缩放）；
* 分阶段冻结（规格 §9）与三组学习率（thermal 1e-4 / action expert 5e-5 /
  PaliGemma 1e-5）；
* 梯度裁剪、按 ``save_interval`` 保存检查点、``--resume`` 断点续训；
* 热觉辅助损失与 flow loss 联合反传，分项损失写日志（可选 wandb）。

用法::

    python -m training.pi05.train --config configs/training/pi05_thermal_stage1.yaml

    # 断点续训 / 覆盖重训
    python -m training.pi05.train --config ... --resume
    python -m training.pi05.train --config ... --overwrite
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
import dataclasses
import logging
import pathlib
import platform
import shutil
import time

import numpy as np
import safetensors.torch
import torch

from models.pi05.observation import Observation
from models.pi05.pi0_pytorch import PI0Pytorch, load_state_dict_compat
from training.pi05.thermal.finetune import configure_stage, parameter_groups, stage_loss_keys
from training.pi05.thermal.labels import build_thermal_supervision
from training.pi05.thermal.step import thermal_train_step
from training.pi05.checkpoints import latest_checkpoint_step, load_checkpoint, save_checkpoint
from training.pi05.config import TrainConfig, load_train_config
from training.pi05.data import build_dataloader

logger = logging.getLogger("thermal_pi05")


# --------------------------------------------------------------------------- #
# 基础设施
# --------------------------------------------------------------------------- #
def init_logging() -> None:
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class _Formatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = _Formatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s (%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        root.addHandler(handler)
    else:
        root.handlers[0].setFormatter(formatter)


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def make_lr_schedule(config: TrainConfig):
    """官方语义：warmup 从 peak/(warmup+1) 线性升到 peak，再余弦衰减到 decay_lr。"""
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    def lr_schedule(step: int) -> float:
        if step < warmup_steps:
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    return lr_schedule


def _infinite(loader) -> Iterator[dict]:
    while True:
        yield from loader


def _is_checkpoint_dir(directory: pathlib.Path) -> bool:
    """只允许删除看起来像检查点目录的路径（数字目录、tmp_*、norm_stats.json）。"""
    for child in directory.iterdir():
        if child.is_dir() and (child.name.isdigit() or child.name.startswith("tmp_")):
            continue
        if child.is_file() and child.name in ("norm_stats.json", "wandb_id.txt"):
            continue
        return False
    return True


def _init_wandb(config: TrainConfig, *, resuming: bool):
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - 取决于运行环境
        raise RuntimeError("配置开启了 wandb，但当前环境没有安装 wandb（pip install wandb）") from exc
    if resuming:
        run_id = (config.checkpoint_path / "wandb_id.txt").read_text(encoding="utf-8").strip()
        return wandb.init(id=run_id, resume="must", project=config.wandb_project)
    run = wandb.init(name=config.exp_name, config=dataclasses.asdict(config), project=config.wandb_project)
    (config.checkpoint_path / "wandb_id.txt").write_text(run.id, encoding="utf-8")
    return run


# --------------------------------------------------------------------------- #
# 主循环
# --------------------------------------------------------------------------- #
def train_loop(config: TrainConfig) -> None:
    device = resolve_device(config.device)
    set_seed(config.seed)
    logger.info("设备：%s | 主机：%s", device, platform.node())

    # ---- 检查点目录 -------------------------------------------------------
    checkpoint_dir = config.checkpoint_path
    resuming = config.resume
    if resuming:
        if latest_checkpoint_step(checkpoint_dir) is None:
            raise FileNotFoundError(f"{checkpoint_dir} 下没有可续训的检查点")
        logger.info("从 %s 续训，最新 step=%s", checkpoint_dir, latest_checkpoint_step(checkpoint_dir))
    elif config.overwrite and checkpoint_dir.exists():
        if not _is_checkpoint_dir(checkpoint_dir):
            raise RuntimeError(f"{checkpoint_dir} 含有非检查点文件，拒绝 --overwrite 删除")
        shutil.rmtree(checkpoint_dir)
        logger.info("已清空旧检查点目录：%s", checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if config.wandb_enabled:
        wandb_run = _init_wandb(config, resuming=resuming)

    # ---- 数据 -------------------------------------------------------------
    loader, normalizer = build_dataloader(config, shuffle=True)
    logger.info(
        "数据：kind=%s 批大小=%d 样本数=%d 图像键=%s",
        config.data.kind,
        config.data.batch_size,
        len(loader.dataset),
        config.data.image_keys,
    )

    # ---- 模型 -------------------------------------------------------------
    model = PI0Pytorch(config.model).to(device)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("已启用梯度检查点")

    if config.pytorch_weight_path is not None:
        weights = pathlib.Path(config.pytorch_weight_path) / "model.safetensors"
        if not weights.exists():
            raise FileNotFoundError(f"找不到预训练权重：{weights}")
        load_state_dict_compat(model, safetensors.torch.load_file(weights, device=str(device)))
        logger.info("已加载预训练权重：%s", weights)

    thermal_stage = config.thermal_stage
    thermal_loss_keys = stage_loss_keys(thermal_stage) if thermal_stage is not None else None
    if thermal_stage is not None:
        counts = configure_stage(model, thermal_stage)
        logger.info("热觉阶段 %d：可训练参数 %s，反传损失项 %s", thermal_stage, counts, thermal_loss_keys)

    # ---- 优化器 -----------------------------------------------------------
    peak_lr = config.lr_schedule.peak_lr
    if thermal_stage is not None:
        groups = parameter_groups(model, thermal_stage)
        for group in groups:
            group["lr_scale"] = float(group["lr"]) / peak_lr
        optimizer = torch.optim.AdamW(
            groups,
            lr=peak_lr,
            betas=(config.optimizer.b1, config.optimizer.b2),
            eps=config.optimizer.eps,
            weight_decay=config.optimizer.weight_decay,
        )
        logger.info("参数组：%s", {group["name"]: f"{group['lr']:.1e}" for group in groups})
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=peak_lr,
            betas=(config.optimizer.b1, config.optimizer.b2),
            eps=config.optimizer.eps,
            weight_decay=config.optimizer.weight_decay,
        )

    global_step = 0
    if resuming:
        global_step = load_checkpoint(
            model=model, optimizer=optimizer, checkpoint_dir=checkpoint_dir, device=device
        )
        logger.info("已恢复训练状态：step=%d", global_step)

    # ---- 循环 -------------------------------------------------------------
    lr_schedule = make_lr_schedule(config)
    model.train()
    batches = _infinite(loader)
    infos: list[dict] = []
    start_time = time.time()
    logger.info(
        "开始训练：num_train_steps=%d batch_size=%d dtype=%s weight_decay=%g clip_norm=%g",
        config.num_train_steps,
        config.data.batch_size,
        config.dtype,
        config.optimizer.weight_decay,
        config.optimizer.clip_gradient_norm,
    )

    try:
        while global_step < config.num_train_steps:
            batch = _to_device(next(batches), device)
            supervision_arrays = batch.pop("thermal_supervision", None)
            actions = batch.pop("actions")
            observation = Observation.from_dict(batch)

            for group in optimizer.param_groups:
                group["lr"] = lr_schedule(global_step) * group.get("lr_scale", 1.0)

            loss_terms: dict[str, torch.Tensor] = {}
            if config.model.thermal is not None and config.model.thermal.enabled:
                supervision = (
                    build_thermal_supervision(supervision_arrays, device=device)
                    if supervision_arrays is not None
                    else None
                )
                loss, loss_terms = thermal_train_step(
                    model, observation, actions, supervision, loss_keys=thermal_loss_keys
                )
            else:
                loss = model(observation, actions).mean()

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=config.optimizer.clip_gradient_norm,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            infos.append(
                {
                    "loss": float(loss.detach()),
                    "lr": optimizer.param_groups[0]["lr"],
                    "grad_norm": float(grad_norm),
                    "terms": {name: float(value) for name, value in loss_terms.items()},
                }
            )

            if global_step % config.log_interval == 0 or global_step == config.num_train_steps:
                elapsed = time.time() - start_time
                count = len(infos)
                avg_loss = sum(info["loss"] for info in infos) / count
                avg_lr = sum(info["lr"] for info in infos) / count
                avg_grad_norm = sum(info["grad_norm"] for info in infos) / count
                term_names = sorted({name for info in infos for name in info["terms"]})
                avg_terms = {
                    name: sum(info["terms"].get(name, 0.0) for info in infos) / count for name in term_names
                }
                logger.info(
                    "step=%d loss=%.4f lr=%.2e grad_norm=%.2f %.2fs/step",
                    global_step,
                    avg_loss,
                    avg_lr,
                    avg_grad_norm,
                    elapsed / max(1, count),
                )
                if avg_terms:
                    logger.info("  " + " ".join(f"{name}={value:.4f}" for name, value in avg_terms.items()))
                if wandb_run is not None:
                    payload = {
                        "loss": avg_loss,
                        "learning_rate": avg_lr,
                        "grad_norm": avg_grad_norm,
                        "step": global_step,
                        "time_per_step": elapsed / max(1, count),
                    }
                    payload.update({f"thermal/{name}": value for name, value in avg_terms.items()})
                    wandb_run.log(payload, step=global_step)
                infos = []
                start_time = time.time()

            saved = save_checkpoint(
                model=model,
                optimizer=optimizer,
                global_step=global_step,
                config=config,
                normalizer=normalizer,
                final=global_step == config.num_train_steps,
            )
            if saved is not None and wandb_run is not None:
                wandb_run.log({"checkpoint_step": global_step}, step=global_step)
    except KeyboardInterrupt:
        logger.warning("收到中断，正在保存当前检查点（step=%d）", global_step)
        if global_step > 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                global_step=global_step,
                config=config,
                normalizer=normalizer,
                final=True,
            )
        raise
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    logger.info("训练完成：共 %d 步，检查点目录 %s", global_step, checkpoint_dir)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Thermal-π0.5 训练/微调")
    parser.add_argument("--config", required=True, help="训练配置 YAML 路径")
    parser.add_argument("--resume", action="store_true", help="从最新检查点续训")
    parser.add_argument("--overwrite", action="store_true", help="删除旧检查点后从头训练")
    parser.add_argument("--checkpoint-dir", default=None, help="覆盖配置里的 checkpoint_dir")
    parser.add_argument("--device", default=None, help="覆盖配置里的 device（auto/cpu/cuda）")
    parser.add_argument("--max-steps", type=int, default=None, help="覆盖 num_train_steps（冒烟用）")
    parser.add_argument("--wandb", dest="wandb", action="store_true", default=None, help="强制开启 wandb")
    parser.add_argument("--no-wandb", dest="wandb", action="store_false", help="强制关闭 wandb")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    config = load_train_config(args.config)
    overrides = {}
    if args.resume:
        overrides["resume"] = True
    if args.overwrite:
        overrides["overwrite"] = True
    if args.checkpoint_dir is not None:
        overrides["checkpoint_dir"] = args.checkpoint_dir
    if args.device is not None:
        overrides["device"] = args.device
    if args.max_steps is not None:
        overrides["num_train_steps"] = args.max_steps
    if args.wandb is not None:
        overrides["wandb_enabled"] = args.wandb
    return dataclasses.replace(config, **overrides)


def main(argv: list[str] | None = None) -> None:
    init_logging()
    config = config_from_args(parse_args(argv))
    train_loop(config)


if __name__ == "__main__":
    main()

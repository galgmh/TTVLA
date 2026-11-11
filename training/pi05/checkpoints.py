"""检查点读写（参考 openpi `scripts/train_pytorch.py` 的 save/load_checkpoint）。

目录布局::

    <checkpoint_dir>/
        3000/
            model.safetensors     # 模型权重（safetensors.save_model）
            optimizer.pt          # AdamW 状态
            metadata.pt           # global_step / config / timestamp
            norm_stats.json       # state/action 归一化统计
        norm_stats.json           # 最新一份（推理侧直接读取）

写入先落到 ``tmp_<step>/`` 再原子改名为 ``<step>/``，中途失败不会留下半个
检查点。最新检查点按目录名的整数大小判定。
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
import shutil
import time

import safetensors.torch
import torch

from models.pi05.pi0_pytorch import load_state_dict_compat
from training.pi05.config import TrainConfig
from training.pi05.transforms import Normalizer

logger = logging.getLogger(__name__)


def latest_checkpoint_step(checkpoint_dir: str | pathlib.Path) -> int | None:
    """返回最新的步数目录名（没有则 None）。"""
    directory = pathlib.Path(checkpoint_dir)
    if not directory.is_dir():
        return None
    steps = [
        int(child.name)
        for child in directory.iterdir()
        if child.is_dir() and child.name.isdigit() and not child.name.startswith("tmp_")
    ]
    return max(steps) if steps else None


def save_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    config: TrainConfig,
    normalizer: Normalizer,
    final: bool = False,
) -> pathlib.Path | None:
    """按 ``save_interval`` / 最后一步保存；未到点返回 None。"""
    due = (global_step % config.save_interval == 0 and global_step > 0) or final
    if not due:
        return None

    checkpoint_dir = config.checkpoint_path
    final_dir = checkpoint_dir / str(global_step)
    tmp_dir = checkpoint_dir / f"tmp_{global_step}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    safetensors.torch.save_model(model, tmp_dir / "model.safetensors")
    torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt")
    torch.save(
        {
            "global_step": global_step,
            "config": dataclasses.asdict(config),
            "timestamp": time.time(),
        },
        tmp_dir / "metadata.pt",
    )
    normalizer.save(tmp_dir / "norm_stats.json")
    normalizer.save(checkpoint_dir / "norm_stats.json")

    if final_dir.exists():
        shutil.rmtree(final_dir)
    tmp_dir.rename(final_dir)
    logger.info("检查点已保存：step=%d -> %s", global_step, final_dir)
    return final_dir


def _alias_tied_weights(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """补齐 safetensors 去重时丢掉的共享权重名（Gemma 的 embed_tokens / lm_head）。

    ``safetensors.save_model`` 只写一份共享存储，因此加载时会少一个名字；
    模型里这两个参数本来就指向同一块存储，只要把已加载的张量按别名补进去，
    ``load_state_dict`` 就会原地更新共享参数，不会真正复制数据。
    """
    missing = [name for name in model.state_dict() if name not in state_dict]
    if not missing:
        return state_dict
    params = dict(model.state_dict())
    filled = dict(state_dict)
    for name in missing:
        target = params[name]
        for loaded_name, loaded in state_dict.items():
            if params[loaded_name].data_ptr() == target.data_ptr():
                filled[name] = loaded
                break
    return filled


def load_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    checkpoint_dir: str | pathlib.Path,
    device: torch.device | str,
) -> int:
    """加载最新检查点，返回 ``global_step``。"""
    directory = pathlib.Path(checkpoint_dir)
    step = latest_checkpoint_step(directory)
    if step is None:
        raise FileNotFoundError(f"{directory} 下没有检查点")
    ckpt = directory / str(step)

    state_dict = safetensors.torch.load_file(ckpt / "model.safetensors", device=str(device))
    state_dict = _alias_tied_weights(model, state_dict)
    load_state_dict_compat(model, state_dict)

    if optimizer is not None:
        optimizer_path = ckpt / "optimizer.pt"
        if not optimizer_path.exists():
            raise FileNotFoundError(f"{ckpt} 缺少 optimizer.pt")
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=device, weights_only=True))

    metadata_path = ckpt / "metadata.pt"
    if not metadata_path.exists():
        raise FileNotFoundError(f"{ckpt} 缺少 metadata.pt")
    metadata = torch.load(metadata_path, map_location="cpu", weights_only=True)
    return int(metadata.get("global_step", step))


__all__ = ["latest_checkpoint_step", "load_checkpoint", "save_checkpoint"]

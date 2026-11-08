"""模型配置加载（严格键检查，未知键直接报错）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_MODEL_KEYS = {
    "in_channels",
    "hidden",
    "levels",
    "kernel",
    "dropout",
    "out_dim",
}
_TRAINING_KEYS = {
    "batch_size",
    "learning_rate",
    "weight_decay",
    "epochs",
    "seed",
    "weights",
    "output_dir",
    "device",
    "num_workers",
}
_DATA_KEYS = {
    "root",
    "train",
    "validation",
    "test_id",
    "test_finger_history",
    "test_sensor_shift",
    "test_mixed_contact",
}
_WEIGHT_KEYS = {"temp", "valid"}


def _check_keys(mapping: Any, required: set[str], where: str) -> None:
    if not isinstance(mapping, dict):
        raise ValueError(f"{where} 必须是映射，实为 {type(mapping).__name__}")
    missing = required - set(mapping)
    unknown = set(mapping) - required
    if missing:
        raise ValueError(f"{where} 缺少键：{sorted(missing)}")
    if unknown:
        raise ValueError(f"{where} 出现未知键：{sorted(unknown)}")


def load_model_config(path: str | Path) -> dict[str, Any]:
    """读取模型/训练/数据配置（三段，键集合固定）。"""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    _check_keys(cfg, {"model", "training", "data"}, str(path))
    _check_keys(cfg["model"], _MODEL_KEYS, f"{path}: model")
    _check_keys(cfg["training"], _TRAINING_KEYS, f"{path}: training")
    _check_keys(cfg["training"]["weights"], _WEIGHT_KEYS, f"{path}: training.weights")
    _check_keys(cfg["data"], _DATA_KEYS, f"{path}: data")
    return cfg


__all__ = ["load_model_config"]

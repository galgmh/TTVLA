"""Thermal-π0.5 训练配置。

参考 openpi `src/openpi/training/config.py` 的组织方式重写：
``TrainConfig`` 由 YAML 构造，模型配置直接复用 `models.pi05.pi0_config.Pi0Config`，
热觉分支配置复用 `models.pi05.thermal.config.ThermalPi0Config`。
加载时严格拒绝未知键，避免 YAML 拼写错误被静默忽略。
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import pathlib
from typing import Any

import yaml

from models.pi05.pi0_config import Pi0Config
from models.pi05.thermal.config import ThermalPi0Config

#: 训练精度（PyTorch dtype 字符串）。
DTYPES = ("float32", "bfloat16", "float16")

#: 允许的数据后端：fake 仅用于冒烟/单元测试。
DATA_KINDS = ("fake", "hdf5")


@dataclasses.dataclass(frozen=True)
class OptimizerConfig:
    """AdamW 超参数（与官方 PyTorch 训练器的默认值一致）。"""

    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 1e-10
    clip_gradient_norm: float = 1.0


@dataclasses.dataclass(frozen=True)
class LRScheduleConfig:
    """warmup + cosine decay，语义与官方 train_pytorch.py 相同。"""

    warmup_steps: int = 1000
    peak_lr: float = 1e-4
    decay_steps: int = 30000
    decay_lr: float = 1e-5


@dataclasses.dataclass(frozen=True)
class DataConfig:
    """数据与批次配置。

    ``kind="hdf5"`` 时 ``root`` 指向一集一个 HDF5 文件的目录（格式见
    `training.pi05.data` 模块文档）；``kind="fake"`` 时生成随机样本，只用于
    冒烟与单元测试。

    ``action_dim`` 是数据集/机器人实际动作维度（LIBERO 为 7），不是 π0.5
    内部动作维度。state 保留数据集自己的维度（LIBERO 为 8）。模型内部始终
    保持官方 32 维；数据变换先按各自原始维度完成归一化，再补零到
    ``model.action_dim``。``image_keys`` 必须包含 π0.5 的
    三路标准相机键（``base_0_rgb`` / ``left_wrist_0_rgb`` /
    ``right_wrist_0_rgb``）。
    """

    kind: str = "fake"
    root: str | None = None
    batch_size: int = 8
    num_workers: int = 0
    image_keys: tuple[str, ...] = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    prompt: str = "perform the thermal manipulation task"
    action_dim: int = 7
    action_horizon: int = 50
    normalization: str | None = None
    fake_size: int = 128

    def __post_init__(self):
        if self.kind not in DATA_KINDS:
            raise ValueError(f"data.kind 必须是 {DATA_KINDS} 之一，得到 {self.kind!r}")
        if self.kind == "hdf5" and not self.root:
            raise ValueError("data.kind=hdf5 时必须提供 data.root")
        if not self.image_keys:
            raise ValueError("data.image_keys 不能为空")
        for name, value in (
            ("batch_size", self.batch_size),
            ("action_dim", self.action_dim),
            ("action_horizon", self.action_horizon),
            ("fake_size", self.fake_size),
        ):
            if value < 1:
                raise ValueError(f"data.{name} 必须为正整数，得到 {value}")


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    """一次训练/微调运行的完整配置。"""

    exp_name: str
    model: Pi0Config
    data: DataConfig
    optimizer: OptimizerConfig
    lr_schedule: LRScheduleConfig
    seed: int = 0
    dtype: str = "bfloat16"
    num_train_steps: int = 30000
    log_interval: int = 10
    save_interval: int = 1000
    checkpoint_dir: str = "checkpoints/pi05_thermal"
    resume: bool = False
    overwrite: bool = False
    wandb_enabled: bool = False
    wandb_project: str = "thermal-pi05"
    pytorch_weight_path: str | None = None
    thermal_contact_checkpoint: str | None = None
    thermal_stage: int | None = None
    device: str = "auto"

    def __post_init__(self):
        if self.dtype not in DTYPES:
            raise ValueError(f"dtype 必须是 {DTYPES} 之一，得到 {self.dtype!r}")
        if self.thermal_stage is not None and self.thermal_stage not in (1, 2, 3, 4, 5):
            raise ValueError(f"thermal_stage 必须是 1-5 或 null，得到 {self.thermal_stage!r}")
        if self.thermal_stage is not None:
            thermal = self.model.thermal
            if thermal is None or not thermal.enabled:
                raise ValueError("设置 thermal_stage 时必须开启 model.thermal.enabled")
        if self.data.action_dim > self.model.action_dim:
            raise ValueError(
                f"机器人 data.action_dim={self.data.action_dim} 不能超过模型内部 "
                f"model.action_dim={self.model.action_dim}"
            )
        if self.data.action_horizon != self.model.action_horizon:
            raise ValueError(
                f"data.action_horizon={self.data.action_horizon} 与 model.action_horizon="
                f"{self.model.action_horizon} 不一致"
            )
        for name, value in (
            ("num_train_steps", self.num_train_steps),
            ("log_interval", self.log_interval),
            ("save_interval", self.save_interval),
        ):
            if value < 1:
                raise ValueError(f"{name} 必须为正整数，得到 {value}")

    @property
    def checkpoint_path(self) -> pathlib.Path:
        return pathlib.Path(self.checkpoint_dir)


# --------------------------------------------------------------------------- #
# YAML -> TrainConfig
# --------------------------------------------------------------------------- #

_TOP_KEYS = {field.name for field in dataclasses.fields(TrainConfig)}
_MODEL_KEYS = {
    "dtype",
    "paligemma_variant",
    "action_expert_variant",
    "action_dim",
    "action_horizon",
    "max_token_len",
    "pi05",
    "discrete_state_input",
    "pytorch_compile_mode",
    "thermal",
}
_DATA_KEYS = {field.name for field in dataclasses.fields(DataConfig)}
_OPTIMIZER_KEYS = {field.name for field in dataclasses.fields(OptimizerConfig)}
_SCHEDULE_KEYS = {field.name for field in dataclasses.fields(LRScheduleConfig)}
_THERMAL_KEYS = {field.name for field in dataclasses.fields(ThermalPi0Config)}


def _check_keys(section: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ValueError(f"{name} 含未知键 {unknown}；允许的键：{sorted(allowed)}")


def _section(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} 必须是映射，得到 {type(value).__name__}")
    return value


def _build_model(raw: Mapping[str, Any], dtype: str) -> Pi0Config:
    _check_keys(raw, _MODEL_KEYS, "model")
    thermal_raw = raw.get("thermal")
    thermal = None
    if thermal_raw is not None:
        if not isinstance(thermal_raw, Mapping):
            raise ValueError("model.thermal 必须是映射")
        _check_keys(thermal_raw, _THERMAL_KEYS, "model.thermal")
        thermal = ThermalPi0Config(**thermal_raw)
    model_keys = {k: v for k, v in raw.items() if k != "thermal"}
    model_keys.setdefault("dtype", dtype)
    if model_keys["dtype"] != dtype:
        raise ValueError(f"model.dtype={model_keys['dtype']!r} 与顶层 dtype={dtype!r} 不一致")
    return Pi0Config(thermal=thermal, **model_keys)


def load_train_config(path: str | pathlib.Path) -> TrainConfig:
    """从 YAML 读取并校验训练配置。"""
    raw = yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path} 的内容必须是 YAML 映射")
    # ``wandb`` 是唯一允许的嵌套小节（映射到 wandb_enabled / wandb_project）。
    _check_keys(raw, _TOP_KEYS | {"wandb"}, "训练配置")

    dtype = raw.get("dtype", "bfloat16")
    model = _build_model(_section(raw, "model"), dtype)
    data_raw = _section(raw, "data")
    _check_keys(data_raw, _DATA_KEYS, "data")
    if "image_keys" in data_raw:
        data_raw = {**data_raw, "image_keys": tuple(data_raw["image_keys"])}
    data = DataConfig(**data_raw)

    optimizer_raw = _section(raw, "optimizer")
    _check_keys(optimizer_raw, _OPTIMIZER_KEYS, "optimizer")
    optimizer = OptimizerConfig(**optimizer_raw)

    schedule_raw = _section(raw, "lr_schedule")
    _check_keys(schedule_raw, _SCHEDULE_KEYS, "lr_schedule")
    schedule = LRScheduleConfig(**schedule_raw)

    train_keys = {
        k: v
        for k, v in raw.items()
        if k not in ("model", "data", "optimizer", "lr_schedule")
    }
    wandb_raw = train_keys.pop("wandb", None) or {}
    if not isinstance(wandb_raw, Mapping):
        raise ValueError("wandb 必须是映射")
    _check_keys(wandb_raw, {"enabled", "project"}, "wandb")
    train_keys["wandb_enabled"] = bool(wandb_raw.get("enabled", False))
    train_keys["wandb_project"] = str(wandb_raw.get("project", "thermal-pi05"))

    return TrainConfig(model=model, data=data, optimizer=optimizer, lr_schedule=schedule, **train_keys)


__all__ = [
    "DATA_KINDS",
    "DTYPES",
    "DataConfig",
    "LRScheduleConfig",
    "OptimizerConfig",
    "TrainConfig",
    "load_train_config",
]

"""Thermal-π0.5 训练数据：数据集、样本变换、collate 与 DataLoader。

数据格式 ``thermalvla-1.0``（一集一个 HDF5，由仿真侧导出器生成）::

    episode.hdf5
      attrs:
        language_instruction   (可选，字节/字符串；缺省用 config.data.prompt)
      images/<camera_key>      [T, H, W, 3] uint8
      state                    [T, S] float32
      actions                  [T, ah, ad] float32
      thermal/                 13 个 Observation 热觉字段（逐帧）
        thermal_history            [T, K, L, C] float32
        thermal_history_time_mask  [T, K, L] bool
        contact_mask               [T, K] bool
        contact_xyz_obj            [T, K, 3] float32
        contact_normal_obj         [T, K, 3] float32
        contact_age_s              [T, K, 1] float32
        surface_xyz_obj            [T, N, 3] float32
        surface_normal_obj         [T, N, 3] float32
        surface_mask               [T, N] bool
        surface_reachable          [T, N] bool
        surface_grasp_score        [T, N, 1] float32
        ee_xyz_obj                 [T, 3] float32
        probe_budget_remaining     [T, 1] float32
      supervision/             7 个特权监督字段（只进损失，不进 Observation）
        contact_temp_c             [T, K, 1] float32
        contact_valid              [T, K, 1] float32
        surface_temp_c             [T, N, 1] float32
        surface_temp_mask          [T, N] bool
        surface_safe               [T, N, 1] float32
        strategy_type              [T] int64（0=OTHER, 1=PROBE, 2=GRASP）
        target_surface_index       [T] int64（OTHER 时为 -1）

``FakeThermalVlaDataset`` 生成形状合法的随机样本，只用于冒烟测试和单元测试。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import functools
import logging
import pathlib
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from models.pi05.preprocessing_pytorch import IMAGE_KEYS as MODEL_IMAGE_KEYS
from models.pi05.thermal.config import ThermalPi0Config
from training.pi05.thermal.transforms import FLOAT_KEYS as THERMAL_FLOAT_KEYS
from training.pi05.thermal.transforms import THERMAL_OBSERVATION_KEYS
from training.pi05.config import DataConfig, TrainConfig
from training.pi05.tokenizer import PaligemmaTokenizer
from training.pi05.transforms import Normalizer, pad_to_dim, prepare_sample

logger = logging.getLogger(__name__)

#: 监督字段及其 dtype（bool / 整数项单独处理）。
SUPERVISION_FIELDS = (
    "contact_temp_c",
    "contact_valid",
    "surface_temp_c",
    "surface_temp_mask",
    "surface_safe",
    "strategy_type",
    "target_surface_index",
)
_SUPERVISION_BOOL = frozenset({"surface_temp_mask"})
_SUPERVISION_INT = frozenset({"strategy_type", "target_surface_index"})


# --------------------------------------------------------------------------- #
# 数据集协议
# --------------------------------------------------------------------------- #
class ThermalVlaDataset(Dataset):
    """数据集协议：``__getitem__`` 返回未归一化的单样本字典。

    必需键：``image``（dict[str, uint8 NHWC]）、``state``、``actions``、``prompt``。
    热觉开启时还需 13 个 Observation 字段与 ``thermal_supervision`` 子字典。
    """

    def __len__(self) -> int:  # pragma: no cover - 协议声明
        raise NotImplementedError

    def __getitem__(self, index: int) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def stats(self) -> tuple[np.ndarray, np.ndarray] | None:
        """返回 ``(states, actions)`` 原始数组用于计算归一化统计；None 表示用恒等变换。"""
        return None


class FakeThermalVlaDataset(ThermalVlaDataset):
    """随机样本（固定种子、与 index 一一对应）。仅用于冒烟/单元测试。"""

    def __init__(
        self,
        *,
        size: int,
        image_keys: Sequence[str],
        image_size: int,
        action_dim: int,
        action_horizon: int,
        prompt: str,
        thermal: ThermalPi0Config | None = None,
        seed: int = 0,
    ):
        self._size = size
        self._image_keys = tuple(image_keys)
        self._image_size = image_size
        self._action_dim = action_dim
        self._action_horizon = action_horizon
        self._prompt = prompt
        self._thermal = thermal if thermal is not None and thermal.enabled else None
        self._seed = seed

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = np.random.default_rng((self._seed, index))
        h = w = self._image_size
        sample: dict[str, Any] = {
            "image": {key: rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for key in self._image_keys},
            "state": rng.uniform(-1, 1, self._action_dim).astype(np.float32),
            "actions": rng.uniform(-1, 1, (self._action_horizon, self._action_dim)).astype(np.float32),
            "prompt": self._prompt,
        }
        thermal = self._thermal
        if thermal is None:
            return sample
        k, length, channels, n = thermal.history_size, thermal.window_size, thermal.signal_dim, thermal.num_surface_points
        sample.update(
            {
                "thermal_history": rng.uniform(-5, 45, (k, length, channels)).astype(np.float32),
                "thermal_history_time_mask": rng.random((k, length)) > 0.2,
                "contact_mask": rng.random(k) > 0.3,
                "contact_xyz_obj": rng.uniform(-1, 1, (k, 3)).astype(np.float32),
                "contact_normal_obj": _unit_vectors(rng, k).astype(np.float32),
                "contact_age_s": rng.uniform(0, 10, (k, 1)).astype(np.float32),
                "surface_xyz_obj": rng.uniform(-1, 1, (n, 3)).astype(np.float32),
                "surface_normal_obj": _unit_vectors(rng, n).astype(np.float32),
                "surface_mask": rng.random(n) > 0.1,
                "surface_reachable": rng.random(n) > 0.2,
                "surface_grasp_score": rng.uniform(0, 1, (n, 1)).astype(np.float32),
                "ee_xyz_obj": rng.uniform(-1, 1, 3).astype(np.float32),
                "probe_budget_remaining": np.asarray([rng.uniform(0, 1)], dtype=np.float32),
                "thermal_supervision": {
                    "contact_temp_c": rng.uniform(-5, 45, (k, 1)).astype(np.float32),
                    "contact_valid": (rng.random((k, 1)) > 0.3).astype(np.float32),
                    "surface_temp_c": rng.uniform(-5, 45, (n, 1)).astype(np.float32),
                    "surface_temp_mask": rng.random(n) > 0.1,
                    "surface_safe": rng.random((n, 1)).astype(np.float32),
                    "strategy_type": np.int64(index % 3),
                    "target_surface_index": np.int64(index % n),
                },
            }
        )
        return sample


def _unit_vectors(rng: np.random.Generator, count: int) -> np.ndarray:
    vectors = rng.normal(size=(count, 3))
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


class Hdf5ThermalVlaDataset(ThermalVlaDataset):
    """读取 ``thermalvla-1.0`` 目录（一集一个 ``*.hdf5``）。"""

    def __init__(
        self,
        root: str | pathlib.Path,
        *,
        prompt: str,
        thermal: ThermalPi0Config | None = None,
        camera_keys: Sequence[str] | None = None,
    ):
        self._root = pathlib.Path(root)
        files = sorted(self._root.glob("*.hdf5"))
        if not files:
            raise FileNotFoundError(f"{self._root} 下没有 *.hdf5 文件")
        self._files = files
        self._prompt = prompt
        self._thermal = thermal if thermal is not None and thermal.enabled else None
        self._camera_keys = tuple(camera_keys) if camera_keys is not None else None

        # 索引表：file_id、帧号与帧数；同时读取语言指令（可选）。
        self._index: list[tuple[int, int]] = []
        self._prompts: list[str] = []
        for file_id, path in enumerate(files):
            with h5py.File(path, "r") as f:
                frames = _length(f)
                instruction = f.attrs.get("language_instruction")
                if isinstance(instruction, bytes):
                    instruction = instruction.decode("utf-8")
                prompts = str(instruction).strip() if instruction else prompt
                if self._camera_keys is None:
                    self._camera_keys = tuple(sorted(f["images"].keys()))
                if self._thermal is not None:
                    missing = [key for key in THERMAL_OBSERVATION_KEYS if key not in f["thermal"]]
                    if missing:
                        raise ValueError(f"{path} 缺少热觉字段 {missing}")
            self._index.extend((file_id, frame) for frame in range(frames))
            self._prompts.append(prompts)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        file_id, frame = self._index[index]
        path = self._files[file_id]
        with h5py.File(path, "r") as f:
            sample: dict[str, Any] = {
                "image": {key: np.asarray(f["images"][key][frame], dtype=np.uint8) for key in self._camera_keys},
                "state": np.asarray(f["state"][frame], dtype=np.float32),
                "actions": np.asarray(f["actions"][frame], dtype=np.float32),
                "prompt": self._prompts[file_id],
            }
            if self._thermal is not None:
                for key in THERMAL_OBSERVATION_KEYS:
                    value = np.asarray(f["thermal"][key][frame])
                    sample[key] = value.astype(bool) if key not in THERMAL_FLOAT_KEYS else value.astype(np.float32)
                sample["thermal_supervision"] = {
                    key: np.asarray(f["supervision"][key][frame]) for key in SUPERVISION_FIELDS
                }
        return sample

    def stats(self) -> tuple[np.ndarray, np.ndarray] | None:
        """分块统计全部帧的 state / action（不必把整集读进内存）。"""
        states: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        chunk = 1024
        for path in self._files:
            with h5py.File(path, "r") as f:
                frames = _length(f)
                for start in range(0, frames, chunk):
                    stop = min(start + chunk, frames)
                    states.append(np.asarray(f["state"][start:stop], dtype=np.float32))
                    actions.append(np.asarray(f["actions"][start:stop], dtype=np.float32))
        if not states:
            return None
        return np.concatenate(states), np.concatenate(actions)


def _length(f: h5py.File) -> int:
    for key in ("state", "actions"):
        if key in f:
            return int(f[key].shape[0])
    raise ValueError("HDF5 缺少 state/actions，无法确定帧数")


# --------------------------------------------------------------------------- #
# 变换包装与 collate
# --------------------------------------------------------------------------- #
class _TransformedDataset(Dataset):
    def __init__(self, dataset: ThermalVlaDataset, transform: Callable[[dict[str, Any]], dict[str, Any]]):
        self._dataset = dataset
        self._transform = transform

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._transform(self._dataset[index])


def collate_samples(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """把逐样本字典堆成批次张量；``thermal_supervision`` 保持嵌套字典。"""
    first = samples[0]
    batch: dict[str, Any] = {}

    image_keys = list(first["image"])
    batch["image"] = {
        key: torch.from_numpy(np.stack([np.asarray(s["image"][key]) for s in samples])) for key in image_keys
    }
    batch["image_mask"] = {
        key: torch.tensor([bool(s["image_mask"][key]) for s in samples], dtype=torch.bool) for key in image_keys
    }
    batch["state"] = torch.from_numpy(np.stack([np.asarray(s["state"]) for s in samples])).to(torch.float32)
    batch["actions"] = torch.from_numpy(np.stack([np.asarray(s["actions"]) for s in samples])).to(torch.float32)
    batch["tokenized_prompt"] = torch.from_numpy(
        np.stack([np.asarray(s["tokenized_prompt"]) for s in samples])
    ).to(torch.int64)
    batch["tokenized_prompt_mask"] = torch.from_numpy(
        np.stack([np.asarray(s["tokenized_prompt_mask"]) for s in samples])
    ).to(torch.bool)

    thermal_present = all(key in first for key in THERMAL_OBSERVATION_KEYS)
    if thermal_present:
        for key in THERMAL_OBSERVATION_KEYS:
            values = torch.from_numpy(np.stack([np.asarray(s[key]) for s in samples]))
            batch[key] = values.to(torch.bool) if key not in THERMAL_FLOAT_KEYS else values.to(torch.float32)

    if "thermal_supervision" in first:
        supervision: dict[str, torch.Tensor] = {}
        for key in SUPERVISION_FIELDS:
            values = torch.from_numpy(np.stack([np.asarray(s["thermal_supervision"][key]) for s in samples]))
            if key in _SUPERVISION_BOOL:
                supervision[key] = values.to(torch.bool)
            elif key in _SUPERVISION_INT:
                supervision[key] = values.to(torch.int64)
            else:
                supervision[key] = values.to(torch.float32)
        batch["thermal_supervision"] = supervision
    return batch


# --------------------------------------------------------------------------- #
# 组装
# --------------------------------------------------------------------------- #
def create_dataset(cfg: DataConfig, thermal: ThermalPi0Config | None) -> ThermalVlaDataset:
    if cfg.kind == "fake":
        return FakeThermalVlaDataset(
            size=cfg.fake_size,
            image_keys=cfg.image_keys,
            image_size=128,
            action_dim=cfg.action_dim,
            action_horizon=cfg.action_horizon,
            prompt=cfg.prompt,
            thermal=thermal,
        )
    return Hdf5ThermalVlaDataset(
        cfg.root,  # type: ignore[arg-type]  # DataConfig 已校验 hdf5 必须提供 root
        prompt=cfg.prompt,
        thermal=thermal,
        camera_keys=cfg.image_keys,
    )


def resolve_normalizer(cfg: DataConfig, dataset: ThermalVlaDataset) -> Normalizer:
    """``data.normalization`` 指向 JSON 时直接读取；否则（fake 以外）现场统计。"""
    if cfg.normalization:
        return Normalizer.load(cfg.normalization)
    if cfg.kind == "fake":
        return Normalizer.identity(cfg.action_dim, cfg.action_dim)
    arrays = dataset.stats()
    if arrays is None:
        raise ValueError("数据集没有可用于归一化统计的 state/actions")
    # 与 prepare_sample 一致：state/action 都截到 action_dim 后再统计。
    states, actions = arrays
    states = pad_to_dim(states, cfg.action_dim)
    actions = pad_to_dim(actions, cfg.action_dim)
    logger.info("正在从数据集统计 state/action 归一化参数（%d 帧）", states.shape[0])
    return Normalizer.compute(states, actions)


def build_dataloader(
    cfg: TrainConfig,
    *,
    shuffle: bool = True,
) -> tuple[DataLoader, Normalizer]:
    """返回 ``(DataLoader, Normalizer)``；批次是含 ``thermal_supervision`` 的字典。"""
    # 模型的 `_flow_loss` 内固定按三路标准相机键调用 `preprocess_observation_pytorch`，
    # 数据侧少任何一路都会在深层报错，这里提前给出明确信息。
    missing = [key for key in MODEL_IMAGE_KEYS if key not in cfg.data.image_keys]
    if missing:
        raise ValueError(f"data.image_keys 缺少 π0.5 必需的相机 {missing}（需要 {MODEL_IMAGE_KEYS}）")
    thermal = cfg.model.thermal
    dataset = create_dataset(cfg.data, thermal)
    normalizer = resolve_normalizer(cfg.data, dataset)
    tokenizer = PaligemmaTokenizer(max_len=cfg.model.max_token_len or 200)
    transform = functools.partial(
        prepare_sample,
        normalizer=normalizer,
        tokenizer=tokenizer,
        image_keys=cfg.data.image_keys,
        action_dim=cfg.data.action_dim,
        action_horizon=cfg.data.action_horizon,
    )
    loader = DataLoader(
        _TransformedDataset(dataset, transform),
        batch_size=cfg.data.batch_size,
        shuffle=shuffle,
        num_workers=cfg.data.num_workers,
        drop_last=True,
        persistent_workers=cfg.data.num_workers > 0,
        collate_fn=collate_samples,
    )
    return loader, normalizer


__all__ = [
    "SUPERVISION_FIELDS",
    "FakeThermalVlaDataset",
    "Hdf5ThermalVlaDataset",
    "ThermalVlaDataset",
    "build_dataloader",
    "collate_samples",
    "create_dataset",
    "resolve_normalizer",
]

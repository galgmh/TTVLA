"""单样本变换：归一化、维度补齐、π0.5 指令分词。

参考 openpi `src/openpi/transforms.py` 中的 `Normalize` / `PadStatesAndActions` /
`TokenizePrompt`，但把统计量改成项目自己的 mean/std（数据层一个 JSON），
并直接在 numpy 样本字典上运行，便于 DataLoader 多进程。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import dataclasses
import json
import pathlib
from typing import Any

import numpy as np

from training.pi05.tokenizer import PaligemmaTokenizer


@dataclasses.dataclass(frozen=True)
class Normalizer:
    """state / action 的逐维归一化统计（mean、std）。"""

    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    def __post_init__(self):
        for name in ("state_mean", "state_std", "action_mean", "action_std"):
            value = np.asarray(getattr(self, name), dtype=np.float32)
            if value.ndim != 1:
                raise ValueError(f"normalizer.{name} 必须是一维数组")
            if name.endswith("_std") and np.any(value <= 0):
                raise ValueError(f"normalizer.{name} 必须全为正数")
        object.__setattr__(self, "state_mean", np.asarray(self.state_mean, dtype=np.float32))
        object.__setattr__(self, "state_std", np.asarray(self.state_std, dtype=np.float32))
        object.__setattr__(self, "action_mean", np.asarray(self.action_mean, dtype=np.float32))
        object.__setattr__(self, "action_std", np.asarray(self.action_std, dtype=np.float32))

    # ------------------------------------------------------------------ #
    # 构造 / 读写
    # ------------------------------------------------------------------ #
    @classmethod
    def identity(cls, state_dim: int, action_dim: int) -> Normalizer:
        return cls(
            state_mean=np.zeros(state_dim, np.float32),
            state_std=np.ones(state_dim, np.float32),
            action_mean=np.zeros(action_dim, np.float32),
            action_std=np.ones(action_dim, np.float32),
        )

    @classmethod
    def compute(cls, states: np.ndarray, actions: np.ndarray) -> Normalizer:
        """用数据集里的全部 state / action 计算 mean/std（std 下限 1e-6）。"""

        def stats(values: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
            values = np.asarray(values, dtype=np.float32).reshape(-1, values.shape[-1])
            if values.shape[0] == 0:
                raise ValueError(f"{name} 为空，无法计算归一化统计")
            mean = values.mean(axis=0)
            std = values.std(axis=0)
            std = np.maximum(std, 1e-6)
            return mean, std

        state_mean, state_std = stats(states, "state")
        action_mean, action_std = stats(actions, "action")
        return cls(state_mean, state_std, action_mean, action_std)

    @classmethod
    def load(cls, path: str | pathlib.Path) -> Normalizer:
        raw = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        return cls(
            np.asarray(raw["state_mean"], np.float32),
            np.asarray(raw["state_std"], np.float32),
            np.asarray(raw["action_mean"], np.float32),
            np.asarray(raw["action_std"], np.float32),
        )

    def save(self, path: str | pathlib.Path) -> None:
        pathlib.Path(path).write_text(
            json.dumps(
                {
                    "state_mean": self.state_mean.tolist(),
                    "state_std": self.state_std.tolist(),
                    "action_mean": self.action_mean.tolist(),
                    "action_std": self.action_std.tolist(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------ #
    # 变换
    # ------------------------------------------------------------------ #
    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        return ((np.asarray(state, np.float32) - self.state_mean) / self.state_std).astype(np.float32)

    def normalize_actions(self, actions: np.ndarray) -> np.ndarray:
        return ((np.asarray(actions, np.float32) - self.action_mean) / self.action_std).astype(np.float32)

    def denormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        return (np.asarray(actions, np.float32) * self.action_std + self.action_mean).astype(np.float32)


def pad_or_trim(values: np.ndarray, length: int, *, axis: int = 0) -> np.ndarray:
    """沿 ``axis`` 把数组补齐（重复最后一帧）或截断到 ``length``。"""
    values = np.asarray(values)
    current = values.shape[axis]
    if current == length:
        return values
    if current > length:
        return np.take(values, np.arange(length), axis=axis)
    pad_shape = list(values.shape)
    pad_shape[axis] = length - current
    if current == 0:
        pad = np.zeros(pad_shape, dtype=values.dtype)
    else:
        last = np.take(values, [current - 1], axis=axis)
        pad = np.repeat(last, length - current, axis=axis)
    return np.concatenate([values, pad], axis=axis)


def pad_to_dim(values: np.ndarray, dim: int) -> np.ndarray:
    """把最后一维补齐/截断到 ``dim``（补 0）。"""
    values = np.asarray(values, dtype=np.float32)
    current = values.shape[-1]
    if current == dim:
        return values
    if current > dim:
        return values[..., :dim]
    pad = np.zeros((*values.shape[:-1], dim - current), dtype=np.float32)
    return np.concatenate([values, pad], axis=-1)


def tokenize_sample(sample: Mapping[str, Any], tokenizer: PaligemmaTokenizer) -> tuple[np.ndarray, np.ndarray]:
    """用已归一化的 state 生成 π0.5 的语言 token 与 mask。"""
    return tokenizer.tokenize(str(sample["prompt"]), state=np.asarray(sample["state"], np.float32))


def prepare_sample(
    sample: Mapping[str, Any],
    *,
    normalizer: Normalizer,
    tokenizer: PaligemmaTokenizer,
    image_keys: Iterable[str],
    action_dim: int,
    action_horizon: int,
) -> dict[str, Any]:
    """把数据集样本整理成模型输入（单样本，无 batch 维）。

    - 图像保持 uint8 NHWC；`Observation.from_dict` 会转成 NCHW float [-1, 1]，
      分辨率再由 `preprocess_observation_pytorch` 统一 resize/pad 到 224。
    - state 与 action 都补齐/截断到 ``action_dim``（与官方 `PadStatesAndActions`
      一致，热觉编码器的 state_proj 按 action_dim 建层），再用 mean/std 归一化。
    - π0.5 的离散状态来自**归一化后**的 state，必须在这之后分词。
    """
    output: dict[str, Any] = {}
    images = sample["image"]
    missing = [key for key in image_keys if key not in images]
    if missing:
        raise KeyError(f"样本缺少图像 {missing}；已有 {sorted(images)}")
    output["image"] = {key: np.asarray(images[key], dtype=np.uint8) for key in image_keys}
    output["image_mask"] = {key: True for key in image_keys}

    state = pad_to_dim(sample["state"], action_dim)
    output["state"] = normalizer.normalize_state(state)
    actions = pad_to_dim(sample["actions"], action_dim)
    actions = pad_or_trim(actions, action_horizon, axis=0)
    output["actions"] = normalizer.normalize_actions(actions)

    tokens, token_mask = tokenize_sample({"prompt": sample["prompt"], "state": output["state"]}, tokenizer)
    output["tokenized_prompt"] = tokens
    output["tokenized_prompt_mask"] = token_mask

    for key, value in sample.items():
        if key in ("image", "state", "actions", "prompt", "image_mask"):
            continue
        output[key] = value
    return output


__all__ = [
    "Normalizer",
    "pad_or_trim",
    "pad_to_dim",
    "prepare_sample",
    "tokenize_sample",
]

"""惰性窗口数据集：按文件缓存 ``WindowSet``，向外只暴露单样本。

正式训练集 2 万集，一次性把所有窗口建进内存要上 GB。这里只在第一次
访问某个文件时构建该文件的窗口，用 ``OrderedDict`` 保留最近若干个文件
（训练时 DataLoader 打乱窗口顺序，命中率足够）。

``data/datasets/thermal_contact_dataset.py`` 负责“怎么切窗口”，本模块只
负责“从哪个文件取第几个窗口”，两边共用 :func:`window_end_times`，不会
出现两套切法。
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from data.datasets.thermal_contact_dataset import (
    WindowSet,
    apply_stats,
    build_windows,
    load_episode,
    window_end_times,
)

DEFAULT_MIN_CONTACT_S = 0.05
DEFAULT_WINDOW_STRIDE_S = 0.05
DEFAULT_NO_CONTACT_WINDOW_S = 0.5


def count_windows(
    path: str | Path,
    *,
    prefixes_s: Sequence[float],
    stride: int,
    min_contact_s: float = DEFAULT_MIN_CONTACT_S,
    window_stride_s: float = DEFAULT_WINDOW_STRIDE_S,
    no_contact_window_s: float = DEFAULT_NO_CONTACT_WINDOW_S,
) -> int:
    """只读时间和接触判定，算出一个文件能切多少个窗口。"""
    import h5py

    with h5py.File(path, "r") as handle:
        time = np.asarray(handle["time/sim_time"][...])[::stride]
        mask = np.asarray(handle["obs/contact_mask"][...]).astype(bool)[::stride]
    ends = window_end_times(
        {"time/sim_time": time, "obs/contact_mask": mask},
        min_contact_s=min_contact_s,
        window_stride_s=window_stride_s,
        no_contact_window_s=no_contact_window_s,
    )
    return int(len(ends) * len(prefixes_s))


class ThermalWindowDataset(Dataset):
    """``files`` 里所有 episode 的窗口拼成一个数据集。"""

    def __init__(
        self,
        files: Sequence[str | Path],
        *,
        prefixes_s: Sequence[float],
        training_hz: float,
        stride: int,
        stats: dict[str, list[float]] | None = None,
        window_stride_s: float = DEFAULT_WINDOW_STRIDE_S,
        min_contact_s: float = DEFAULT_MIN_CONTACT_S,
        no_contact_window_s: float = DEFAULT_NO_CONTACT_WINDOW_S,
        cache_size: int = 8,
    ):
        if cache_size < 1:
            raise ValueError("cache_size 必须 >= 1")
        self.files = [Path(f) for f in files]
        self.prefixes_s = tuple(float(p) for p in prefixes_s)
        self.training_hz = float(training_hz)
        self.stride = int(stride)
        self.stats = stats
        self.window_stride_s = float(window_stride_s)
        self.min_contact_s = float(min_contact_s)
        self.no_contact_window_s = float(no_contact_window_s)
        self.cache_size = int(cache_size)

        self.counts = np.asarray(
            [
                count_windows(
                    f,
                    prefixes_s=self.prefixes_s,
                    stride=self.stride,
                    min_contact_s=self.min_contact_s,
                    window_stride_s=self.window_stride_s,
                    no_contact_window_s=self.no_contact_window_s,
                )
                for f in self.files
            ],
            dtype=np.int64,
        )
        self.offsets = np.concatenate([[0], np.cumsum(self.counts)])
        self._cache: OrderedDict[int, WindowSet] = OrderedDict()

    # ------------------------------------------------------------ Dataset 协议
    def __len__(self) -> int:
        return int(self.offsets[-1])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if not 0 <= index < len(self):
            raise IndexError(f"窗口下标 {index} 超出 [0, {len(self)})")
        file_index = int(np.searchsorted(self.offsets, index, side="right") - 1)
        local = int(index - self.offsets[file_index])
        windows = self._file_windows(file_index)
        return {
            "features": torch.from_numpy(windows.features[local]),
            "mask": torch.from_numpy(windows.mask[local]),
            "temperature": torch.tensor(float(windows.temperature[local])),
            "contact_valid": torch.tensor(bool(windows.contact_valid[local])),
        }

    # ------------------------------------------------------------------ 内部
    def _file_windows(self, file_index: int) -> WindowSet:
        cached = self._cache.get(file_index)
        if cached is not None:
            self._cache.move_to_end(file_index)
            return cached
        windows = build_windows(
            load_episode(self.files[file_index]),
            prefixes_s=self.prefixes_s,
            training_hz=self.training_hz,
            stride=self.stride,
            window_stride_s=self.window_stride_s,
            min_contact_s=self.min_contact_s,
            no_contact_window_s=self.no_contact_window_s,
        )
        if self.stats is not None:
            windows = apply_stats(windows, self.stats)
        self._cache[file_index] = windows
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return windows

    def file_index_of(self, index: int) -> int:
        """窗口下标 -> 文件下标（排查用）。"""
        return int(np.searchsorted(self.offsets, index, side="right") - 1)


def compute_stats(
    files: Sequence[str | Path],
    *,
    prefixes_s: Sequence[float],
    training_hz: float,
    stride: int,
    window_stride_s: float = DEFAULT_WINDOW_STRIDE_S,
    min_contact_s: float = DEFAULT_MIN_CONTACT_S,
    no_contact_window_s: float = DEFAULT_NO_CONTACT_WINDOW_S,
    max_files: int | None = None,
) -> dict[str, list[float]]:
    """在训练集上流式累计逐通道均值/标准差（只统计有 mask 的帧）。"""
    selected = list(files)[: max_files] if max_files else list(files)
    if not selected:
        raise ValueError("没有训练文件，无法计算归一化统计")
    total = 0
    total_sum = None
    total_sq = None
    for path in selected:
        windows = build_windows(
            load_episode(path),
            prefixes_s=prefixes_s,
            training_hz=training_hz,
            stride=stride,
            window_stride_s=window_stride_s,
            min_contact_s=min_contact_s,
            no_contact_window_s=no_contact_window_s,
        )
        if windows.features.shape[0] == 0:
            continue
        data = windows.features[windows.mask].astype(np.float64)
        if data.size == 0:
            continue
        if total_sum is None:
            total_sum = data.sum(axis=0)
            total_sq = (data**2).sum(axis=0)
        else:
            total_sum += data.sum(axis=0)
            total_sq += (data**2).sum(axis=0)
        total += data.shape[0]
    if not total or total_sum is None:
        raise ValueError("训练窗口全为空，无法计算归一化统计")
    mean = total_sum / total
    var = np.maximum(total_sq / total - mean**2, 0.0)
    std = np.sqrt(var)
    std = np.where(std < 1e-8, 1.0, std)
    return {"mean": mean.tolist(), "std": std.tolist()}


__all__ = ["ThermalWindowDataset", "compute_stats", "count_windows"]

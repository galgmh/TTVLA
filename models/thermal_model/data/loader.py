"""读取已经生成的 ``thermal-encoder-v3`` 窗口和计算归一化统计。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PreparedThermalEncoderDataset(Dataset):
    """直接读取 ``thermal-encoder-v3`` 派生窗口 HDF5。"""

    def __init__(self, path: str | Path, *, stats: dict[str, list[float]] | None = None):
        import h5py

        self.path = Path(path)
        with h5py.File(self.path, "r") as handle:
            self.format = str(handle.attrs.get("format", ""))
            if self.format != "thermal-encoder-v3":
                raise ValueError(
                    f"{self.path} 不是 thermal-encoder-v3 数据；"
                    "请运行 thermal_model.encoder_pretraining.prepare_dataset --overwrite 重建"
                )
            self.training_hz = float(handle.attrs["training_hz"])
            self.length = int(handle["features"].shape[0])
            self.window_size = int(handle["features"].shape[1])
            self.channels = int(handle["features"].shape[2])
            if self.channels != 12:
                raise ValueError(f"{self.path} 特征通道应为 12，实为 {self.channels}")
        self.stats = stats
        if stats is not None:
            if len(stats["mean"]) != self.channels or len(stats["std"]) != self.channels:
                raise ValueError("归一化统计的通道数与派生数据不一致")
        self._handle = None

    def __len__(self) -> int:
        return self.length

    def _file(self):
        import h5py

        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if not 0 <= index < self.length:
            raise IndexError(f"窗口下标 {index} 超出 [0, {self.length})")
        handle = self._file()
        features = np.asarray(handle["features"][index], dtype=np.float32)
        mask = np.asarray(handle["mask"][index], dtype=bool)
        if self.stats is not None:
            mean = np.asarray(self.stats["mean"], dtype=np.float32)
            std = np.asarray(self.stats["std"], dtype=np.float32)
            features = (features - mean) / std
            features[~mask] = 0.0
        return {
            "features": torch.from_numpy(features),
            "mask": torch.from_numpy(mask),
            "temperature": torch.tensor(float(handle["temperature"][index])),
            "contact_valid": torch.tensor(bool(handle["contact_valid"][index])),
        }

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_handle"] = None
        return state

    def __del__(self):
        if self._handle is not None:
            try:
                self._handle.close()
            except Exception:
                pass


class PreparedSpatialDataset(Dataset):
    """读取 ``thermal-spatial-v2`` 空间热觉预训练数据。"""

    def __init__(self, path: str | Path, *, stats: dict[str, list[float]] | None = None):
        import h5py

        self.path = Path(path)
        with h5py.File(self.path, "r") as handle:
            self.format = str(handle.attrs.get("format", ""))
            if self.format != "thermal-spatial-v2":
                raise ValueError(
                    f"{self.path} 不是 thermal-spatial-v2 数据；"
                    "请运行 thermal_model.spatial_pretraining.prepare_dataset --overwrite"
                )
            self.training_hz = float(handle.attrs["training_hz"])
            self.length = int(handle["features"].shape[0])
            self.window_size = int(handle["features"].shape[1])
            self.channels = int(handle["features"].shape[2])
            self.query_count = int(handle["query_xyz"].shape[1])
        self.stats = stats
        if stats is not None:
            if len(stats["mean"]) != self.channels or len(stats["std"]) != self.channels:
                raise ValueError("归一化统计的通道数与空间预训练数据不一致")
        self._handle = None

    def __len__(self) -> int:
        return self.length

    def _file(self):
        import h5py

        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if not 0 <= index < self.length:
            raise IndexError(f"窗口下标 {index} 超出 [0, {self.length})")
        handle = self._file()
        features = np.asarray(handle["features"][index], dtype=np.float32)
        mask = np.asarray(handle["mask"][index], dtype=bool)
        if self.stats is not None:
            mean = np.asarray(self.stats["mean"], dtype=np.float32)
            std = np.asarray(self.stats["std"], dtype=np.float32)
            features = (features - mean) / std
            features[~mask] = 0.0
        return {
            "features": torch.from_numpy(features),
            "mask": torch.from_numpy(mask),
            "positions": torch.from_numpy(
                np.asarray(handle["positions"][index], dtype=np.float32)
            ),
            "normals": torch.from_numpy(
                np.asarray(handle["normals"][index], dtype=np.float32)
            ),
            "query_xyz": torch.from_numpy(
                np.asarray(handle["query_xyz"][index], dtype=np.float32)
            ),
            "query_temperature": torch.from_numpy(
                np.asarray(handle["query_temperature"][index], dtype=np.float32)
            ),
            "query_mask": torch.from_numpy(
                np.asarray(handle["query_mask"][index], dtype=bool)
            ),
        }

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_handle"] = None
        return state

    def __del__(self):
        if self._handle is not None:
            try:
                self._handle.close()
            except Exception:
                pass


def compute_prepared_stats(path: str | Path, *, chunk_size: int = 4096) -> dict[str, list[float]]:
    """只用派生训练文件中 mask=True 的帧计算逐通道统计。"""
    import h5py

    total = 0
    total_sum = None
    total_sq = None
    with h5py.File(path, "r") as handle:
        count = int(handle["features"].shape[0])
        for start in range(0, count, chunk_size):
            stop = min(count, start + chunk_size)
            features = np.asarray(handle["features"][start:stop], dtype=np.float64)
            mask = np.asarray(handle["mask"][start:stop], dtype=bool)
            data = features[mask]
            if not data.size:
                continue
            summed = data.sum(axis=0)
            squared = (data**2).sum(axis=0)
            total_sum = summed if total_sum is None else total_sum + summed
            total_sq = squared if total_sq is None else total_sq + squared
            total += int(data.shape[0])
    if total == 0 or total_sum is None:
        raise ValueError(f"{path} 没有可用于统计的有效帧")
    mean = total_sum / total
    std = np.sqrt(np.maximum(total_sq / total - mean**2, 0.0))
    std = np.where(std < 1e-8, 1.0, std)
    return {"mean": mean.tolist(), "std": std.tolist()}


__all__ = [
    "PreparedSpatialDataset",
    "PreparedThermalEncoderDataset",
    "compute_prepared_stats",
]

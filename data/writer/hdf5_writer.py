"""HDF5 写出器：一份文件一集，只写原始量。

目录结构见 :mod:`data.schema.thermal_contact_v1`。写出器不做任何派生计算
（不写 ΔT、不重采样、不归一化）——那是加载器的事；也不改字段顺序或名字，
校验器按同一份契约读取。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from data.schema.thermal_contact_v1 import (
    FORMAT_VERSION,
    FRAME_FIELDS,
    SCALAR_FIELDS,
    stack_episode,
)


def _ensure_groups(handle) -> None:
    for name in ("time", "obs", "action", "privileged"):
        if name not in handle:
            handle.create_group(name)


def write_episode(
    path: str | Path,
    buffers: Mapping[str, list],
    metadata: Mapping[str, Any],
    scalars: Mapping[str, Any],
    *,
    compression: str | None = "gzip",
    compression_opts: int = 4,
) -> Path:
    """把一集的逐帧缓冲与元数据写成 HDF5。

    ``buffers`` 是 ``{字段: list}``（见 ``FRAME_FIELDS``）；``scalars`` 提供
    ``privileged/target_region``。
    """
    import h5py

    arrays = stack_episode(buffers)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as handle:
        handle.attrs["format_version"] = FORMAT_VERSION
        handle.attrs["episode"] = json.dumps(dict(metadata), ensure_ascii=False, sort_keys=True)
        _ensure_groups(handle)

        for name, spec in FRAME_FIELDS.items():
            arr = arrays[name]
            kwargs = {}
            if compression is not None and arr.shape[0] > 64:
                kwargs = {"compression": compression, "compression_opts": compression_opts}
            handle.create_dataset(name, data=arr, dtype=spec.dtype, **kwargs)

        for name, spec in SCALAR_FIELDS.items():
            if name not in scalars:
                raise KeyError(f"缺少标量 {name}")
            value = scalars[name]
            handle.create_dataset(name, data=np.asarray(value, dtype=spec.dtype))

    return path


def read_episode(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, np.ndarray]]:
    """读回一集：``(数组, 元数据, 标量)``。

    写读共用同一份字段表，所以这里不做额外校验——校验交给
    ``data.validation``。
    """
    import h5py

    arrays: dict[str, np.ndarray] = {}
    scalars: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as handle:
        metadata = json.loads(handle.attrs["episode"])
        format_version = str(handle.attrs["format_version"])
        for name in FRAME_FIELDS:
            arrays[name] = np.asarray(handle[name][...])
        for name in SCALAR_FIELDS:
            scalars[name] = np.asarray(handle[name][...])
    metadata["format_version"] = format_version
    return arrays, metadata, scalars


__all__ = ["read_episode", "write_episode"]

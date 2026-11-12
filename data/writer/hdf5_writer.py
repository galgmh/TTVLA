"""按 ThermalContact 多时间轴契约读写单 episode HDF5。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from data.schema.thermal_contact_v1 import (
    EPISODE_METADATA_KEYS,
    FORMAT_VERSION,
    FRAME_FIELDS,
    STATIC_FIELDS,
    stack_episode,
    validate_arrays,
)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def write_episode(
    path: str | Path,
    buffers: Mapping[str, list],
    metadata: Mapping[str, Any],
    static: Mapping[str, Any],
    *,
    compression: str | None = "gzip",
    compression_opts: int = 4,
) -> Path:
    """写出一集；必需 episode 标识直接写 attrs，扩展元数据写 JSON attr。"""
    import h5py

    missing_meta = [key for key in EPISODE_METADATA_KEYS if key not in metadata]
    if missing_meta:
        raise KeyError(f"缺少 episode attrs: {missing_meta}")
    missing_static = [key for key in STATIC_FIELDS if key not in static]
    if missing_static:
        raise KeyError(f"缺少静态数据集: {missing_static}")

    arrays = stack_episode(buffers)
    all_arrays = {**arrays, **{key: np.asarray(static[key]) for key in STATIC_FIELDS}}
    issues = validate_arrays(all_arrays)
    if issues:
        raise ValueError("数据不符合 HDF5 契约:\n- " + "\n- ".join(issues))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["format_version"] = FORMAT_VERSION
        for key in EPISODE_METADATA_KEYS:
            handle.attrs[key] = metadata[key]
        handle.attrs["episode_metadata"] = json.dumps(
            dict(metadata), ensure_ascii=False, sort_keys=True, allow_nan=False,
            default=_json_default,
        )

        for name, spec in FRAME_FIELDS.items():
            arr = arrays[name]
            kwargs: dict[str, Any] = {}
            if compression is not None and arr.size > 0 and (arr.shape[0] > 64 or name.startswith("camera/")):
                kwargs = {"compression": compression, "compression_opts": compression_opts}
                if name.endswith("_rgb"):
                    kwargs["chunks"] = (1, *arr.shape[1:])
            ds = handle.create_dataset(name, data=arr, dtype=spec.dtype, **kwargs)
            ds.attrs["description"] = spec.description

        for name, spec in STATIC_FIELDS.items():
            ds = handle.create_dataset(name, data=np.asarray(static[name], dtype=spec.dtype), dtype=spec.dtype)
            ds.attrs["description"] = spec.description
    return path


def read_episode(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, np.ndarray]]:
    """读回 ``(all_arrays, metadata, static_arrays)``。"""
    import h5py

    arrays: dict[str, np.ndarray] = {}
    static: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as handle:
        raw = handle.attrs.get("episode_metadata", "{}")
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        metadata = json.loads(str(raw))
        for key in EPISODE_METADATA_KEYS:
            value = handle.attrs[key]
            metadata[key] = value.decode("utf-8") if isinstance(value, bytes) else value.item() if isinstance(value, np.generic) else value
        metadata["format_version"] = str(handle.attrs["format_version"])
        for name in FRAME_FIELDS:
            arrays[name] = np.asarray(handle[name][...])
        for name in STATIC_FIELDS:
            value = np.asarray(handle[name][...])
            arrays[name] = value
            static[name] = value
    return arrays, metadata, static


__all__ = ["read_episode", "write_episode"]

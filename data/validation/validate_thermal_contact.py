"""ThermalContact-v2 HDF5 契约与物理一致性校验。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data.schema.thermal_contact_v1 import EPISODE_METADATA_KEYS, FORMAT_VERSION, validate_arrays
from data.writer.hdf5_writer import read_episode

CONTACT_FORCE_EPS_N = 0.05
MAX_PLAUSIBLE_FORCE_N = 20.0


def _check_physics(arrays: dict[str, np.ndarray], metadata: dict) -> list[str]:
    issues: list[str] = []
    normal = np.asarray(arrays["tactile/contact_normal_force"], dtype=np.float64)
    mask = np.asarray(arrays["tactile/contact_mask"], dtype=bool)
    link_mask = np.asarray(arrays["privileged/contact_link_mask"], dtype=bool)
    valid = np.asarray(arrays["privileged/contact_valid"], dtype=bool)
    if not np.array_equal(valid, link_mask.any(axis=2)):
        issues.append("privileged/contact_valid 必须等于 contact_link_mask.any(link)")
    strong = normal > CONTACT_FORCE_EPS_N
    if np.any(strong & ~mask):
        issues.append("存在明显法向力但 tactile/contact_mask 为假的指尖样本")
    if np.nanmax(normal, initial=0.0) > MAX_PLAUSIBLE_FORCE_N:
        issues.append(f"法向力峰值超过 {MAX_PLAUSIBLE_FORCE_N:.0f} N")
    temp = np.asarray(arrays["tactile/fingertip_temp"], dtype=np.float64)
    if not np.all(np.isfinite(temp)) or np.nanmin(temp) < 0.0 or np.nanmax(temp) > 100.0:
        issues.append("指尖温度含非有限值或超出 [0,100] °C")
    link_temp = np.asarray(arrays["privileged/link_temp"], dtype=np.float64)
    if not np.all(np.isfinite(link_temp)):
        issues.append("privileged/link_temp 含非有限值")
    return issues


def _check_rates(arrays: dict[str, np.ndarray]) -> list[str]:
    issues: list[str] = []
    for key, dt_expected in (("sensor/time", 0.01), ("control/time", 0.05)):
        time = np.asarray(arrays[key], dtype=np.float64)
        if time.size > 1:
            dt = np.diff(time)
            if np.max(np.abs(dt - dt_expected)) > 1e-6:
                issues.append(
                    f"{key} 采样步长应为 {dt_expected:.3f}s，"
                    f"实际为 [{dt.min():.6f},{dt.max():.6f}]"
                )
    return issues


def validate_file(path: str | Path) -> list[str]:
    path = Path(path)
    try:
        arrays, metadata, _ = read_episode(path)
    except Exception as exc:
        return [f"{path.name}: 无法读取 ({exc})"]
    issues = validate_arrays(arrays)
    issues += [f"缺少 attr {key}" for key in EPISODE_METADATA_KEYS if key not in metadata]
    if metadata.get("format_version") != FORMAT_VERSION:
        issues.append(f"format_version={metadata.get('format_version')!r}，期望 {FORMAT_VERSION!r}")
    if not issues:
        issues += _check_rates(arrays)
        issues += _check_physics(arrays, metadata)
    try:
        json.dumps(metadata, allow_nan=False)
    except (TypeError, ValueError):
        issues.append("episode 元数据含不可 JSON 序列化值或 NaN/Inf")
    return [f"{path.name}: {issue}" for issue in issues]


def validate_directory(root: str | Path, pattern: str = "**/*.hdf5") -> list[str]:
    root = Path(root)
    files = sorted(root.glob(pattern))
    if not files:
        return [f"{root}: 没有找到任何 {pattern} 文件"]
    return [issue for path in files for issue in validate_file(path)]


__all__ = ["CONTACT_FORCE_EPS_N", "MAX_PLAUSIBLE_FORCE_N", "validate_directory", "validate_file"]

"""ThermalContact-v1 数据校验器。

检查分三层：

* **契约**：字段齐全、形状正确、时间严格递增（``schema.validate_arrays``）；
* **物理**：接触力与接触判定一致、接触段时长与结局一致、无异常大冲击、
  温度在物理范围内；
* **纯度**：``privileged/pure_contact`` 必须与 ``contact_link_mask`` 和
  ``target_region`` 严格对应——纯接触集里绝不能出现跨区域接触。

校验器只读文件，不修数据。任何问题都带 ``文件:字段`` 前缀，便于定位。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data.schema.thermal_contact_v1 import (
    EPISODE_METADATA_KEYS,
    FORMAT_VERSION,
    REGION_ORDER,
    validate_arrays,
)
from data.writer.hdf5_writer import read_episode

# 接触力判定阈值（N）：与 sim.obs.CONTACT_FORCE_EPS_N 保持一致。
CONTACT_FORCE_EPS_N = 0.05
# 冲击上限（N）：超过说明手臂撞了杯子，这集不能用。
MAX_PLAUSIBLE_FORCE_N = 20.0


def _check_physics(arrays: dict[str, np.ndarray], metadata: dict, scalars: dict[str, np.ndarray]) -> list[str]:
    issues: list[str] = []
    normal = np.asarray(arrays["obs/contact_normal_force"], dtype=np.float64)
    mask = np.asarray(arrays["obs/contact_mask"], dtype=bool)
    pure = np.asarray(arrays["privileged/pure_contact"], dtype=bool)
    link_mask = np.asarray(arrays["privileged/contact_link_mask"], dtype=bool)
    target = int(np.asarray(scalars["privileged/target_region"]).reshape(()))

    hit = np.where(mask.any(axis=1))[0]

    # 接触判定与法向力一致：有接触标志却全程零力，或有明显力却没标志。
    if hit.size and np.nanmax(normal[hit]) < CONTACT_FORCE_EPS_N:
        issues.append("接触标志为真但法向力全程接近 0")
    strong = np.where(normal.max(axis=1) > CONTACT_FORCE_EPS_N)[0]
    if strong.size and not mask[strong].any(axis=1).all():
        issues.append("存在明显法向力但 contact_mask 为假的帧")

    if np.nanmax(normal) > MAX_PLAUSIBLE_FORCE_N:
        issues.append(f"法向力峰值 {np.nanmax(normal):.1f} N 超过冲击上限 {MAX_PLAUSIBLE_FORCE_N:.0f} N")

    # 接触有效性必须与接触归属表一致。
    valid = np.asarray(arrays["privileged/contact_valid"], dtype=bool)
    if not np.array_equal(valid, link_mask.any(axis=(1, 2))):
        bad = int(np.sum(valid != link_mask.any(axis=(1, 2))))
        issues.append(f"contact_valid 与 contact_link_mask 不一致（{bad} 帧）")

    # 纯接触标志必须是“接触归属表”的子集：报告为纯接触的帧，必须只碰到
    # 目标区域。反向不成立——外来接触（手/腕碰到杯子）会让帧变脏，但归属
    # 表里仍可能只有目标区域的 link，所以只查单向蕴含。
    if target >= 0:
        for i in np.where(pure)[0]:
            touched = np.where(link_mask[i].any(axis=0))[0]
            if touched.size != 1 or int(touched[0]) != target:
                issues.append(f"第 {int(i)} 帧标为纯接触，但接触归属为 {touched.tolist()}")
                break
    elif pure.any():
        issues.append("混杂接触集不应有 pure_contact 为真的帧")

    # 结局与接触段一致。
    started = metadata.get("contact_start_time")
    ended = metadata.get("contact_end_time")
    if started is None and hit.size:
        issues.append("元数据说没有接触，但序列里出现了接触帧")
    if started is not None and not hit.size:
        issues.append("元数据记录了接触，但序列里没有接触帧")
    if started is not None and ended is not None and ended < started:
        issues.append(f"contact_end_time {ended} 早于 contact_start_time {started}")

    # 温度范围。
    temp = np.asarray(arrays["obs/fingertip_temp"], dtype=np.float64)
    if np.nanmin(temp) < 0.0 or np.nanmax(temp) > 80.0:
        issues.append(f"指尖温度超出生理范围 [{np.nanmin(temp):.1f}, {np.nanmax(temp):.1f}] °C")

    # 区域编号与区域顺序。
    if target >= len(REGION_ORDER):
        issues.append(f"target_region {target} 超出区域数 {len(REGION_ORDER)}")
    if metadata.get("region_order") != list(REGION_ORDER):
        issues.append(f"region_order {metadata.get('region_order')} 与契约 {list(REGION_ORDER)} 不一致")
    return issues


def _check_metadata(metadata: dict, arrays: dict[str, np.ndarray]) -> list[str]:
    issues = [f"缺少元数据 {key}" for key in EPISODE_METADATA_KEYS if key not in metadata]
    if metadata.get("format_version") != FORMAT_VERSION:
        issues.append(f"format_version {metadata.get('format_version')} 与契约 {FORMAT_VERSION} 不一致")

    # 采样频率必须是 100 Hz：时间步长恒为 10 ms。
    t = np.asarray(arrays["time/sim_time"], dtype=np.float64)
    if len(t) > 1:
        dt = np.diff(t)
        if np.max(np.abs(dt - 0.01)) > 1e-6:
            issues.append(f"采样步长应恒为 0.01 s，实际 [{dt.min():.4f}, {dt.max():.4f}]")

    # 目标力与实际峰值：稳定档应当接近目标，失败档应当没有接触。
    quality = metadata.get("contact_quality")
    peak = float(metadata.get("peak_normal_force_n", 0.0))
    target_force = metadata.get("target_force_n")
    if quality == "failed" and peak > CONTACT_FORCE_EPS_N:
        issues.append(f"失败档不应接触，实际峰值 {peak:.2f} N")
    if quality == "stable" and isinstance(target_force, (int, float)):
        if peak < 0.5 * float(target_force):
            issues.append(f"稳定档峰值 {peak:.2f} N 远低于目标 {float(target_force):.2f} N")
    return issues


def validate_file(path: str | Path) -> list[str]:
    """校验一个 HDF5 文件，返回问题列表（空列表 = 通过）。"""
    path = Path(path)
    issues: list[str] = []
    try:
        arrays, metadata, scalars = read_episode(path)
    except Exception as exc:  # 读不出来本身就是最严重的问题
        return [f"{path.name}: 无法读取 ({exc})"]

    issues += [f"{path.name}: {msg}" for msg in validate_arrays(arrays)]
    if issues:
        return issues
    issues += [f"{path.name}: {msg}" for msg in _check_physics(arrays, metadata, scalars)]
    issues += [f"{path.name}: {msg}" for msg in _check_metadata(metadata, arrays)]

    raw = metadata.get("material_parameters")
    if not isinstance(raw, dict) or not raw:
        issues.append(f"{path.name}: material_parameters 为空")
    try:
        json.dumps(metadata, allow_nan=False)
    except ValueError:
        issues.append(f"{path.name}: 元数据含 NaN/Inf，无法序列化")
    return issues


def validate_directory(root: str | Path, pattern: str = "**/*.hdf5") -> list[str]:
    """校验目录下所有 episode 文件。"""
    root = Path(root)
    issues: list[str] = []
    files = sorted(root.glob(pattern))
    if not files:
        return [f"{root}: 没有找到任何 {pattern} 文件"]
    for path in files:
        issues += validate_file(path)
    return issues


__all__ = ["CONTACT_FORCE_EPS_N", "MAX_PLAUSIBLE_FORCE_N", "validate_directory", "validate_file"]

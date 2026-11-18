"""筛选严格完成四次接触的 ThermalContact episode。

默认从 ``datasets/thermal_contact_multi/train`` 读取 HDF5，并把符合以下
条件的 episode 以相对软链接放到 ``datasets/thermal_contact_1/train``：

1. ``termination_reason == multi_contact_complete``；
2. 恰好有四个 contact event；
3. 每个 event 只接触一个热区；
4. 四个 event 的实际区域顺序与元数据 ``contact_order`` 一致。

使用软链接不会复制 HDF5 中体积较大的 RGB。需要独立副本时传
``--mode copy``。入口::

    python -m sim.select_episodes --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.config import REPO_ROOT  # noqa: E402


def _repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _metadata(handle: h5py.File) -> dict:
    raw = handle.attrs.get("episode_metadata", "{}")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(str(raw))


def inspect_episode(path: Path) -> dict | None:
    """返回入选 episode 的清单记录；不符合严格成功条件则返回 ``None``。"""
    with h5py.File(path, "r") as handle:
        metadata = _metadata(handle)
        if metadata.get("termination_reason") != "multi_contact_complete":
            return None

        starts = np.asarray(handle["contact_events/start_sensor_index"], dtype=np.int64)
        ends = np.asarray(handle["contact_events/end_sensor_index_exclusive"], dtype=np.int64)
        if len(starts) != 4 or len(ends) != 4:
            return None

        planned_order = [int(value) for value in metadata.get("contact_order", [])]
        if len(planned_order) != 4:
            return None

        link_mask = handle["privileged/contact_link_mask"]
        actual_order: list[int] = []
        durations_s: list[float] = []
        for start, end in zip(starts, ends):
            if start < 0 or end <= start or end > link_mask.shape[0]:
                return None
            regions = np.flatnonzero(
                np.asarray(link_mask[int(start) : int(end)], dtype=bool).any(axis=(0, 1))
            )
            if regions.size != 1:
                return None
            actual_order.append(int(regions[0]))
            durations_s.append((int(end) - int(start)) * 0.01)

        if actual_order != planned_order:
            return None

        return {
            "file": path.name,
            "episode_id": str(metadata.get("episode_id", path.stem)),
            "contact_order": actual_order,
            "planned_hold_s_by_region": [
                float(value) for value in metadata.get("contact_hold_s_by_region", [])
            ],
            "actual_event_duration_s": durations_s,
            "regime": str(metadata.get("regime", "")),
            "peak_normal_force_n": float(metadata.get("peak_normal_force_n", 0.0)),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="筛选严格完成四次接触的 episode")
    parser.add_argument(
        "--source",
        default="datasets/thermal_contact_multi/train",
        help="源 HDF5 目录",
    )
    parser.add_argument(
        "--out",
        default="datasets/thermal_contact_1/train",
        help="输出目录",
    )
    parser.add_argument(
        "--mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="默认创建相对软链接；copy 会复制完整 HDF5",
    )
    parser.add_argument("--expect", type=int, default=90, help="期望入选数量；不一致则停止")
    parser.add_argument("--dry-run", action="store_true", help="只扫描和打印，不创建输出")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source = _repo_path(args.source).resolve()
    output = _repo_path(args.out)
    files = sorted(source.glob("*.hdf5"))
    if not files:
        raise SystemExit(f"源目录没有 HDF5：{source}")

    selected: list[tuple[Path, dict]] = []
    unreadable: list[tuple[str, str]] = []
    for path in files:
        try:
            record = inspect_episode(path)
        except Exception as exc:  # 单个损坏文件不阻止完成扫描，最终统一报告。
            unreadable.append((path.name, f"{type(exc).__name__}: {exc}"))
            continue
        if record is not None:
            selected.append((path, record))

    print(f"扫描 {len(files)} 集，严格成功 {len(selected)} 集，不可读取 {len(unreadable)} 集")
    if len(selected) != int(args.expect):
        raise SystemExit(f"入选数量 {len(selected)} 与 --expect {args.expect} 不一致，未写出")
    if args.dry_run:
        for path, _ in selected:
            print(path.name)
        return 0

    output.mkdir(parents=True, exist_ok=True)
    for source_path, _ in selected:
        destination = output / source_path.name
        if destination.exists() or destination.is_symlink():
            # 可重复运行：目标已经正确指向/复制自同名源文件时直接跳过。
            if destination.is_symlink() and destination.resolve() == source_path.resolve():
                continue
            if args.mode == "copy" and destination.is_file() and destination.stat().st_size == source_path.stat().st_size:
                continue
            raise SystemExit(f"输出已存在且与源不一致：{destination}")
        if args.mode == "symlink":
            destination.symlink_to(os.path.relpath(source_path, start=output))
        else:
            shutil.copy2(source_path, destination)

    selection = {
        "source": str(source),
        "mode": args.mode,
        "count": len(selected),
        "selection": {
            "termination_reason": "multi_contact_complete",
            "contact_event_count": 4,
            "pure_single_region_events": True,
            "actual_order_matches_contact_order": True,
        },
        "episodes": [record for _, record in selected],
        "unreadable": [{"file": name, "error": error} for name, error in unreadable],
    }
    print(f"完成：{len(selected)} 集 -> {output}")
    print(json.dumps(selection["selection"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""打印一份 episode 文件的结构与关键统计。

用法::

    python scripts/inspect_episode.py datasets/thermal_contact_v1/pilot/<id>.hdf5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.schema.thermal_contact_v1 import PHASE_NAMES  # noqa: E402
from data.writer.hdf5_writer import read_episode  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="检查一份 episode 文件")
    parser.add_argument("path")
    parser.add_argument("--metadata", action="store_true", help="打印完整元数据 JSON")
    args = parser.parse_args()

    arrays, metadata, scalars = read_episode(args.path)
    print(f"文件: {args.path}")
    print(f"格式: {metadata.get('format_version')}  帧数: {len(arrays['time/sim_time'])}")
    t = np.asarray(arrays["time/sim_time"], dtype=np.float64)
    if len(t) > 1:
        print(f"时长: {t[-1] - t[0]:.3f} s  步长: {np.median(np.diff(t)) * 1e3:.2f} ms")

    print("\n元数据:")
    keys = (
        "episode_id", "split", "target_region", "regime", "finger_mode",
        "contact_quality", "contact_hold_s", "target_force_level", "target_force_n",
        "initial_finger_temperature", "link_temperature",
        "contact_start_time", "contact_end_time", "termination_reason",
        "policy_steps", "peak_normal_force_n", "frames", "pure_contact_frames",
        "contact_frames",
    )
    for key in keys:
        if key in metadata:
            print(f"  {key}: {metadata[key]}")
    if args.metadata:
        print(json.dumps(metadata, indent=2, ensure_ascii=False))

    print("\n标量:")
    for name, value in scalars.items():
        print(f"  {name}: {np.asarray(value).reshape(-1).tolist()}")

    print("\n逐帧数据集:")
    for name in sorted(arrays):
        arr = np.asarray(arrays[name])
        if arr.dtype.kind == "f" and arr.size:
            # 用 3 位有效数字：接触面积这类 1e-4 量级的通道不会被四舍五入成 0。
            print(f"  {name:<32} {str(arr.shape):<12} {arr.dtype}  [{np.nanmin(arr):.3g}, {np.nanmax(arr):.3g}]")
        else:
            print(f"  {name:<32} {str(arr.shape):<12} {arr.dtype}")

    phase = np.asarray(arrays["action/phase"]).reshape(-1)
    names = [PHASE_NAMES.get(int(code), str(code)) for code in phase]
    order: list[str] = []
    for name in names:
        if not order or order[-1] != name:
            order.append(name)
    print("\n阶段序列: " + " -> ".join(order))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

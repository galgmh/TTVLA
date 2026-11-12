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

from data.writer.hdf5_writer import read_episode  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="检查一份 episode 文件")
    parser.add_argument("path")
    parser.add_argument("--metadata", action="store_true", help="打印完整元数据 JSON")
    args = parser.parse_args()

    arrays, metadata, static = read_episode(args.path)
    print(f"文件: {args.path}")
    print(
        f"格式: {metadata.get('format_version')}  "
        f"传感器/控制帧: {len(arrays['sensor/time'])}/{len(arrays['control/time'])}"
    )
    t = np.asarray(arrays["sensor/time"], dtype=np.float64)
    if len(t) > 1:
        print(f"时长: {t[-1] - t[0]:.3f} s  步长: {np.median(np.diff(t)) * 1e3:.2f} ms")

    print("\n元数据:")
    keys = (
        "episode_id", "split", "target_region", "regime", "finger_mode",
        "contact_quality", "target_force_n", "initial_finger_temperature",
        "initial_link_temperature", "termination_reason", "policy_steps",
        "peak_normal_force_n", "sensor_frames", "control_frames", "contact_events",
    )
    for key in keys:
        if key in metadata:
            print(f"  {key}: {metadata[key]}")
    if args.metadata:
        print(json.dumps(metadata, indent=2, ensure_ascii=False))

    print("\n静态标定:")
    for name, value in static.items():
        print(f"  {name}: {np.asarray(value).reshape(-1).tolist()}")

    print("\n数据集:")
    for name in sorted(arrays):
        arr = np.asarray(arrays[name])
        if arr.dtype.kind == "f" and arr.size:
            # 用 3 位有效数字：接触面积这类 1e-4 量级的通道不会被四舍五入成 0。
            print(f"  {name:<32} {str(arr.shape):<12} {arr.dtype}  [{np.nanmin(arr):.3g}, {np.nanmax(arr):.3g}]")
        else:
            print(f"  {name:<32} {str(arr.shape):<12} {arr.dtype}")

    starts = arrays["contact_events/start_sensor_index"]
    ends = arrays["contact_events/end_sensor_index_exclusive"]
    print(f"\n接触事件: {list(zip(starts.tolist(), ends.tolist()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

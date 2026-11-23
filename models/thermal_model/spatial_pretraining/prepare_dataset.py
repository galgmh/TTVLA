"""把 thermal-encoder-v3 扩展成空间热觉预训练数据。

输出 ``thermal-spatial-v2``：保留原 26 帧热触觉窗口，并补充每个窗口末端的
左右指垫位置/法向、接触区域、四个热区的 base 坐标查询点和真实温度。
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sim.config import REPO_ROOT, load_task_config  # noqa: E402
from sim.recording import read_metadata  # noqa: E402
from sim.rotations import quat_conj, quat_normalize, quat_rotate  # noqa: E402
from thermal_model.data.loader import compute_prepared_stats  # noqa: E402

FORMAT = "thermal-spatial-v2"
SPLITS = ("train", "validation", "test")

FORMAT_README = """# Spatial Thermal Pretraining Dataset

格式：`thermal-spatial-v2`。每个 split 一个 HDF5。

```text
features             [N,26,12] float32  热触觉窗口
mask                 [N,26]    bool     窗口时间 mask
positions            [N,2,3]   float32  左右指垫中心，base 系
normals              [N,2,3]   float32  左右指垫内法向，base 系
query_xyz            [N,4,3]   float32  四个热区查询点，base 系
query_temperature    [N,4,1]   float32  查询点真实温度
query_mask           [N,4]     bool     查询点监督 mask
query_region         [N,4]     uint8    0 lower / 1 middle / 2 upper / 3 handle
contact_valid        [N]       bool     窗口末端是否接触任一热区
contact_region       [N]       int8     当前接触区域；无接触或混杂接触为-1
episode_index        [N]       int32    原 episode 编号
end_sensor_index     [N]       int32    原 100 Hz sensor 末端下标
episode_offsets      [E+1]     int64    每个 episode 在 N 中的起止偏移
```

每个 episode 内的窗口按 `end_sensor_index` 升序排列。相同
`end_sensor_index` 的重复窗口只保留一条。后续训练可以按
`episode_offsets[e]:episode_offsets[e+1]` 直接切出一条完整序列。

接触编码器使用已有的 `thermal-encoder-v3` checkpoint。空间预训练只训练
ContactTokenBuilder、ThermalSpatialMemory 和训练期辅助 SpatialFieldDecoder。
"""


def _repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _episode_index(path: Path) -> int:
    match = re.search(r"_(\d+)\.hdf5$", path.name)
    if match is None:
        raise ValueError(f"无法从文件名解析 episode index：{path.name}")
    return int(match.group(1))


def _queries_base(metadata: dict, task_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    local = np.asarray(metadata["probe_positions"], dtype=np.float64)
    if local.shape != (4, 3):
        raise ValueError(f"probe_positions 应为 [4,3]，得到 {local.shape}")

    cup_pose = np.asarray(metadata["cup_pose"], dtype=np.float64)
    cup_pos, cup_quat = cup_pose[:3], quat_normalize(cup_pose[3:])
    world = np.stack([cup_pos + quat_rotate(cup_quat, point) for point in local])

    robot = task_cfg["robot"]
    base_pos = np.asarray(robot["base_pos"], dtype=np.float64)
    base_quat_inv = quat_conj(quat_normalize(robot["base_quat"]))
    xyz_base = np.stack(
        [quat_rotate(base_quat_inv, point - base_pos) for point in world]
    )
    return xyz_base.astype(np.float32), np.arange(4, dtype=np.uint8)


def _build_split(
    source: Path,
    raw_files: dict[int, Path],
    task_cfg: dict,
) -> tuple[dict[str, np.ndarray], Counter]:
    with h5py.File(source, "r") as handle:
        features = np.asarray(handle["features"], dtype=np.float32)
        mask = np.asarray(handle["mask"], dtype=bool)
        episode_index = np.asarray(handle["episode_index"], dtype=np.int32)
        end_sensor_index = np.asarray(handle["end_sensor_index"], dtype=np.int32)

    chunks: dict[str, list[np.ndarray]] = {
        "features": [],
        "mask": [],
        "positions": [],
        "normals": [],
        "query_xyz": [],
        "query_temperature": [],
        "query_mask": [],
        "query_region": [],
        "contact_valid": [],
        "contact_region": [],
        "episode_index": [],
        "end_sensor_index": [],
    }
    episode_offsets = [0]
    audit = Counter()

    for episode in np.unique(episode_index):
        rows = np.flatnonzero(episode_index == episode)
        order = np.argsort(end_sensor_index[rows], kind="stable")
        rows = rows[order]
        endpoints = end_sensor_index[rows].astype(np.int64)
        _, first_positions = np.unique(endpoints, return_index=True)
        rows = rows[first_positions]
        endpoints = endpoints[first_positions]
        audit["input_windows"] += int(len(order))
        audit["duplicate_windows"] += int(len(order) - len(rows))

        raw_path = raw_files.get(int(episode))
        if raw_path is None:
            raise ValueError(f"找不到 episode {episode} 对应的原始 HDF5")
        with h5py.File(raw_path, "r") as raw:
            pad_positions = np.asarray(raw["tactile/sensor_position_base"], dtype=np.float32)
            pad_normals = np.asarray(raw["tactile/sensor_normal_base"], dtype=np.float32)
            link_temp = np.asarray(raw["privileged/link_temp"], dtype=np.float32)
            finger_valid = np.asarray(raw["privileged/contact_valid"], dtype=bool)
            contact_links = np.asarray(raw["privileged/contact_link_mask"], dtype=bool)
            queries, regions = _queries_base(read_metadata(raw), task_cfg)

        if np.any(endpoints < 0) or np.any(endpoints >= len(link_temp)):
            raise ValueError(f"{source.name} 中 episode {episode} 的末端下标越界")
        count = len(rows)
        chunks["features"].append(features[rows])
        chunks["mask"].append(mask[rows])
        chunks["positions"].append(pad_positions[endpoints])
        chunks["normals"].append(pad_normals[endpoints])
        chunks["query_xyz"].append(np.repeat(queries[None], count, axis=0))
        chunks["query_temperature"].append(link_temp[endpoints][..., None])
        chunks["query_mask"].append(np.ones((count, 4), dtype=bool))
        chunks["query_region"].append(np.repeat(regions[None], count, axis=0))
        touched = contact_links[endpoints].any(axis=1)
        touched_count = touched.sum(axis=1)
        contact_region = np.full(count, -1, dtype=np.int8)
        single = touched_count == 1
        contact_region[single] = np.argmax(touched[single], axis=1)
        chunks["contact_valid"].append(
            finger_valid[endpoints].any(axis=1)
        )
        chunks["contact_region"].append(contact_region)
        chunks["episode_index"].append(
            np.full(count, int(episode), dtype=np.int32)
        )
        chunks["end_sensor_index"].append(endpoints.astype(np.int32))
        episode_offsets.append(episode_offsets[-1] + count)
        audit["episodes"] += 1
        audit["windows"] += count

    output = {
        name: np.concatenate(values, axis=0)
        for name, values in chunks.items()
    }
    output["episode_offsets"] = np.asarray(episode_offsets, dtype=np.int64)
    return output, audit


def _write(path: Path, rows: dict[str, np.ndarray], attrs: dict) -> None:
    with h5py.File(path, "w") as handle:
        for key, value in attrs.items():
            handle.attrs[key] = value
        for name, value in rows.items():
            compression = "lzf" if value.size else None
            handle.create_dataset(name, data=value, compression=compression)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="构建 thermal-spatial-v2 数据")
    parser.add_argument("--source", default="datasets/thermal_encoder")
    parser.add_argument("--raw-source", default="datasets/thermal_contact_1_copy/train")
    parser.add_argument("--out", default="datasets/thermal_spatial")
    parser.add_argument(
        "--task",
        default="sim/configs/tasks/thermal_safe_grasp.yaml",
        help="用于读取机器人 base pose",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source, raw_source = _repo_path(args.source), _repo_path(args.raw_source)
    output = _repo_path(args.out)
    task_cfg = load_task_config(_repo_path(args.task))
    raw_files = {_episode_index(path): path for path in raw_source.glob("*.hdf5")}
    if not raw_files:
        raise SystemExit(f"原始数据目录没有 HDF5：{raw_source}")

    paths = {split: output / f"{split}.hdf5" for split in SPLITS}
    existing = [path for path in paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"输出已存在：{existing[0]}；确认重建时传 --overwrite")
    if args.overwrite:
        for path in existing:
            path.unlink()
    output.mkdir(parents=True, exist_ok=True)

    reference = source / "train.hdf5"
    if not reference.is_file():
        raise SystemExit(f"缺少编码器训练数据：{reference}")
    with h5py.File(reference, "r") as handle:
        training_hz = float(handle.attrs["training_hz"])

    summary = {"format": FORMAT, "splits": {}, "total": Counter(), "audit": {}}
    for split in SPLITS:
        source_path = source / f"{split}.hdf5"
        if not source_path.is_file():
            raise SystemExit(f"缺少编码器 split：{source_path}")
        rows, audit = _build_split(source_path, raw_files, task_cfg)
        attrs = {
            "format": FORMAT,
            "training_hz": training_hz,
            "window_size": int(rows["features"].shape[1]),
            "channels": int(rows["features"].shape[2]),
            "query_count": int(rows["query_xyz"].shape[1]),
            "sorted_by": "episode_index,end_sensor_index",
            "episode_offsets": "episode_offsets",
            "region_order": json.dumps(
                list(task_cfg["object"]["region_order"]), ensure_ascii=False
            ),
        }
        _write(paths[split], rows, attrs)
        info = {
            "episodes": int(np.unique(rows["episode_index"]).size),
            "windows": int(len(rows["features"])),
            "query_points": int(rows["query_xyz"].shape[1]),
            "removed_duplicates": int(audit["duplicate_windows"]),
        }
        summary["splits"][split] = info
        summary["audit"][split] = dict(audit)
        summary["total"].update(
            {
                "episodes": info["episodes"],
                "windows": info["windows"],
                "removed_duplicates": info["removed_duplicates"],
            }
        )
        print(f"{split}: {info}")

    stats = compute_prepared_stats(paths["train"])
    (output / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    summary["total"] = dict(summary["total"])
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "README.md").write_text(FORMAT_README, encoding="utf-8")
    print(f"完成：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

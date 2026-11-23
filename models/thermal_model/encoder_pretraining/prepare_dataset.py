"""把 ThermalContact episode 派生成固定长度的热编码器训练数据。

每个 episode 产生 4 个接触组和 4 个非接触组。每组连续 51 帧，即 50 Hz
下的 1 s；每组从起点 0..24 切出 25 个长度为 26 的真实窗口。所有窗口都
直接来自源数据，不补零，因此 ``mask`` 全为 True。
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

from thermal_model.preprocessing import frame_features  # noqa: E402
from thermal_model.data.loader import compute_prepared_stats  # noqa: E402
from sim.dataset import ENCODER_INPUT_CHANNELS  # noqa: E402
from sim.config import REPO_ROOT, load_dataset_config  # noqa: E402

FORMAT = "thermal-encoder-v3"
SPLITS = ("train", "validation", "test")
GROUP_FRAMES = 51
WINDOW_LENGTH = 26
WINDOWS_PER_GROUP = 25
CONTACT_GROUPS_PER_EPISODE = 4
NONCONTACT_GROUPS_PER_EPISODE = 4
SAMPLE_NONCONTACT = 0
SAMPLE_PURE_CONTACT = 1
SAMPLE_MIXED_CONTACT = 2

FORMAT_README = """# Thermal Encoder Dataset v3

格式：`thermal-encoder-v3`。`train.hdf5`、`validation.hdf5`、`test.hdf5`
字段相同：

```text
features                    [N,26,12] float32  未归一化热触觉窗口
mask                        [N,26]    bool     全部为 True，窗口无 padding
temperature                 [N]       float32  窗口末端接触 link 温度
contact_valid               [N]       bool     窗口末端是否存在有效接触
sample_kind                 [N]       uint8    0非接触/1纯接触/2混合接触
region                      [N]       int8     接触组区域；混合或非接触为-1
event_id                    [N]       int16    原 episode 事件号；非接触为0
episode_index               [N]       int32    原 episode 编号
group_index                 [N]       int16    本类型内的组号 0..3
window_index                [N]       int16    组内窗口起点 0..24
group_start_sensor_index    [N]       int32    51帧组起点（原100Hz下标）
end_sensor_index            [N]       int32    窗口末端（原100Hz下标）
```

每个接触组把最后一个原始接触帧对齐到最后一个窗口的末端。非接触组
是完整的 51 帧无接触片段。非接触样本的温度标签为 `NaN`，只训练接触
有效性。温度标签只在纯接触窗口末端有限。

每组有 25 个窗口。90 个 episode 合计 720 组、18000 个窗口。
`stats.json` 只使用 train 划分中 `mask=True` 的帧。
划分以 episode 为单位，同一 episode 不会跨 split。
"""


def _repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _episode_index(path: Path) -> int:
    match = re.search(r"_(\d+)\.hdf5$", path.name)
    if match is None:
        raise ValueError(f"无法从文件名解析 episode index：{path.name}")
    return int(match.group(1))


def _split_for(index: int) -> str:
    """每个 (温度场景, 接触顺序) 的12种时长分成10/1/1。"""
    stratum, schedule = divmod(index, 12)
    if schedule == stratum % 12:
        return "validation"
    if schedule == (stratum + 1) % 12:
        return "test"
    return "train"


def _read_arrays(path: Path, stride: int) -> dict[str, np.ndarray]:
    sensor_keys = (
        "sensor/time", "tactile/fingertip_temp", "tactile/contact_force_link",
        "tactile/contact_event_id", "privileged/link_temp",
        "privileged/contact_link_mask", "privileged/contact_valid",
    )
    with h5py.File(path, "r") as handle:
        arrays = {key: np.asarray(handle[key][...])[::stride] for key in sensor_keys}
    return arrays


def _valid_group_starts(inactive: np.ndarray, group_frames: int) -> np.ndarray:
    """返回以该帧开头、连续 group_frames 帧都没有接触的位置。"""
    if inactive.size < group_frames:
        return np.zeros(0, dtype=np.int64)
    counts = np.convolve(
        inactive.astype(np.int32),
        np.ones(group_frames, dtype=np.int32),
        mode="valid",
    )
    return np.flatnonzero(counts == group_frames).astype(np.int64)


def _choose_noncontact_starts(
    inactive: np.ndarray,
    *,
    event_starts: list[int],
    group_frames: int,
) -> list[int]:
    """为每个接触事件选一段此前紧邻的非接触数据。"""
    candidates = _valid_group_starts(inactive, group_frames)
    if candidates.size < NONCONTACT_GROUPS_PER_EPISODE:
        raise ValueError(
            f"只有 {candidates.size} 个完整 {group_frames} 帧非接触组，"
            f"至少需要 {NONCONTACT_GROUPS_PER_EPISODE} 个"
        )

    chosen: list[int] = []
    for event_start in event_starts:
        target = max(0, event_start - group_frames)
        order = np.argsort(np.abs(candidates - target), kind="stable")
        selected = None
        # 先选互不重叠且尽量靠近接触前的组；若空间不足再放宽重叠限制。
        for separate in (True, False):
            for position in order:
                start = int(candidates[position])
                if start in chosen:
                    continue
                if separate and any(abs(start - other) < group_frames for other in chosen):
                    continue
                selected = start
                break
            if selected is not None:
                break
        if selected is None:
            raise ValueError("无法构造足够多的不同非接触组")
        chosen.append(selected)
    return chosen


def _window_rows(
    features: np.ndarray,
    *,
    group_start: int,
    stride: int,
    valid: np.ndarray,
    link_mask: np.ndarray,
    link_temp: np.ndarray,
    sample_kind: int,
    region: int,
    event_id: int,
    group_index: int,
    rows: dict[str, list],
    audit: Counter,
) -> None:
    for window_index in range(WINDOWS_PER_GROUP):
        start = group_start + window_index
        end = start + WINDOW_LENGTH
        endpoint = end - 1
        touched = np.flatnonzero(link_mask[endpoint].any(axis=0))
        endpoint_contact = bool(valid[endpoint])
        temperature = float("nan")
        if endpoint_contact and touched.size == 1:
            temperature = float(link_temp[endpoint, int(touched[0])])

        rows["features"].append(features[start:end])
        rows["mask"].append(np.ones(WINDOW_LENGTH, dtype=bool))
        rows["temperature"].append(temperature)
        rows["contact_valid"].append(endpoint_contact)
        rows["sample_kind"].append(sample_kind)
        rows["region"].append(region)
        rows["event_id"].append(event_id)
        rows["group_index"].append(group_index)
        rows["window_index"].append(window_index)
        rows["group_start_sensor_index"].append(group_start * stride)
        rows["end_sensor_index"].append(endpoint * stride)
        audit["temperature_labels"] += int(np.isfinite(temperature))
        audit["contact_windows"] += int(endpoint_contact)
        audit["noncontact_windows"] += int(not endpoint_contact)


def derive_episode(
    path: Path,
    *,
    training_hz: float,
    stride: int,
) -> tuple[dict[str, np.ndarray], Counter]:
    arrays = _read_arrays(path, stride)
    features = frame_features(arrays, dt=1.0 / training_hz)
    frame_count = len(features)
    if frame_count < GROUP_FRAMES:
        raise ValueError(f"{path.name} 只有 {frame_count} 帧，不足 {GROUP_FRAMES} 帧")

    event_id = np.asarray(arrays["tactile/contact_event_id"], dtype=np.int64)
    valid = np.asarray(arrays["privileged/contact_valid"], dtype=bool).any(axis=1)
    link_mask = np.asarray(arrays["privileged/contact_link_mask"], dtype=bool)
    link_temp = np.asarray(arrays["privileged/link_temp"], dtype=np.float32)

    event_ids = [value for value in np.unique(event_id) if value > 0]
    event_starts: list[int] = []
    event_data: list[tuple[int, np.ndarray]] = []
    for event in event_ids:
        indices = np.flatnonzero(event_id == event)
        if indices.size == 0:
            continue
        event_starts.append(int(indices[0]))
        event_data.append((int(event), indices))
    if len(event_data) < CONTACT_GROUPS_PER_EPISODE:
        raise ValueError(
            f"{path.name} 只有 {len(event_data)} 个接触事件，"
            f"需要 {CONTACT_GROUPS_PER_EPISODE} 个"
        )
    event_data = event_data[:CONTACT_GROUPS_PER_EPISODE]

    rows: dict[str, list] = {
        "features": [],
        "mask": [],
        "temperature": [],
        "contact_valid": [],
        "sample_kind": [],
        "region": [],
        "event_id": [],
        "group_index": [],
        "window_index": [],
        "group_start_sensor_index": [],
        "end_sensor_index": [],
    }
    audit = Counter()

    # 最后一个接触帧对齐到第25个窗口的末端，保证末窗口始终有接触标签。
    for group_index, (event, indices) in enumerate(event_data):
        group_start = int(indices[-1]) - (WINDOWS_PER_GROUP + WINDOW_LENGTH - 2)
        if group_start < 0:
            raise ValueError(f"{path.name} 的事件 {event} 前没有足够帧构造 1 s 组")
        regions = np.flatnonzero(link_mask[indices].any(axis=(0, 1)))
        if regions.size == 1:
            sample_kind, region = SAMPLE_PURE_CONTACT, int(regions[0])
            audit["pure_contact_groups"] += 1
        else:
            sample_kind, region = SAMPLE_MIXED_CONTACT, -1
            audit["mixed_contact_groups"] += 1
        _window_rows(
            features,
            group_start=group_start,
            stride=stride,
            valid=valid,
            link_mask=link_mask,
            link_temp=link_temp,
            sample_kind=sample_kind,
            region=region,
            event_id=event,
            group_index=group_index,
            rows=rows,
            audit=audit,
        )
        audit["contact_groups"] += 1

    inactive = (event_id == 0) & ~valid
    noncontact_starts = _choose_noncontact_starts(
        inactive,
        event_starts=event_starts,
        group_frames=GROUP_FRAMES,
    )
    for group_index, group_start in enumerate(noncontact_starts):
        _window_rows(
            features,
            group_start=group_start,
            stride=stride,
            valid=valid,
            link_mask=link_mask,
            link_temp=link_temp,
            sample_kind=SAMPLE_NONCONTACT,
            region=-1,
            event_id=0,
            group_index=group_index,
            rows=rows,
            audit=audit,
        )
        audit["noncontact_groups"] += 1

    return {
        "features": np.stack(rows["features"]).astype(np.float32),
        "mask": np.stack(rows["mask"]).astype(bool),
        "temperature": np.asarray(rows["temperature"], dtype=np.float32),
        "contact_valid": np.asarray(rows["contact_valid"], dtype=bool),
        "sample_kind": np.asarray(rows["sample_kind"], dtype=np.uint8),
        "region": np.asarray(rows["region"], dtype=np.int8),
        "event_id": np.asarray(rows["event_id"], dtype=np.int16),
        "group_index": np.asarray(rows["group_index"], dtype=np.int16),
        "window_index": np.asarray(rows["window_index"], dtype=np.int16),
        "group_start_sensor_index": np.asarray(
            rows["group_start_sensor_index"], dtype=np.int32
        ),
        "end_sensor_index": np.asarray(rows["end_sensor_index"], dtype=np.int32),
    }, audit


def _create_output(path: Path, attrs: dict) -> h5py.File:
    handle = h5py.File(path, "w")
    for key, value in attrs.items():
        handle.attrs[key] = value
    channels = len(ENCODER_INPUT_CHANNELS)
    handle.create_dataset(
        "features",
        shape=(0, WINDOW_LENGTH, channels),
        maxshape=(None, WINDOW_LENGTH, channels),
        chunks=(256, WINDOW_LENGTH, channels),
        dtype="f4",
        compression="lzf",
    )
    handle.create_dataset(
        "mask",
        shape=(0, WINDOW_LENGTH),
        maxshape=(None, WINDOW_LENGTH),
        chunks=(1024, WINDOW_LENGTH),
        dtype="?",
        compression="lzf",
    )
    for name, dtype in (
        ("temperature", "f4"),
        ("contact_valid", "?"),
        ("sample_kind", "u1"),
        ("region", "i1"),
        ("event_id", "i2"),
        ("episode_index", "i4"),
        ("group_index", "i2"),
        ("window_index", "i2"),
        ("group_start_sensor_index", "i4"),
        ("end_sensor_index", "i4"),
    ):
        handle.create_dataset(
            name,
            shape=(0,),
            maxshape=(None,),
            chunks=(4096,),
            dtype=dtype,
            compression="lzf",
        )
    return handle


def _append(handle: h5py.File, rows: dict[str, np.ndarray], episode_index: int) -> None:
    count = len(rows["temperature"])
    start = int(handle["temperature"].shape[0])
    stop = start + count
    for name, values in rows.items():
        dataset = handle[name]
        dataset.resize(stop, axis=0)
        dataset[start:stop] = values
    dataset = handle["episode_index"]
    dataset.resize(stop, axis=0)
    dataset[start:stop] = episode_index


def _split_summary(path: Path) -> dict:
    with h5py.File(path, "r") as handle:
        episode = np.asarray(handle["episode_index"])
        kind = np.asarray(handle["sample_kind"])
        group = np.asarray(handle["group_index"])
        contact = np.asarray(handle["contact_valid"], dtype=bool)
        temperature = np.asarray(handle["temperature"])
        mask = np.asarray(handle["mask"], dtype=bool)
        groups = set(zip(episode.tolist(), kind.tolist(), group.tolist()))
        contact_groups = {
            key for key in groups if key[1] in (SAMPLE_PURE_CONTACT, SAMPLE_MIXED_CONTACT)
        }
        noncontact_groups = {key for key in groups if key[1] == SAMPLE_NONCONTACT}
        return {
            "episodes": int(np.unique(episode).size),
            "contact_groups": len(contact_groups),
            "noncontact_groups": len(noncontact_groups),
            "groups": len(groups),
            "windows": int(episode.size),
            "contact_windows": int(contact.sum()),
            "noncontact_windows": int((~contact).sum()),
            "temperature_labels": int(np.isfinite(temperature).sum()),
            "mask_all_true": bool(mask.all()),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="构建 thermal-encoder-v3 固定组数据")
    parser.add_argument(
        "--source",
        default="datasets/thermal_contact_1_copy/train",
        help="包含 90 个 ThermalContact episode 的目录",
    )
    parser.add_argument("--out", default="datasets/thermal_encoder")
    parser.add_argument("--dataset", default="sim/generation/presets/thermal_contact_v1.yaml")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source, output = _repo_path(args.source), _repo_path(args.out)
    files = sorted(source.glob("*.hdf5"))
    if not files:
        raise SystemExit(f"源目录没有 HDF5：{source}")

    config = load_dataset_config(_repo_path(args.dataset))
    sample_hz = float(config["sample_hz"])
    training_hz = float(config["training_hz"])
    stride = int(round(sample_hz / training_hz))
    expected_group_frames = int(round(training_hz)) + 1
    if expected_group_frames != GROUP_FRAMES:
        raise SystemExit(
            f"{training_hz:g} Hz 下 1 s 应为 {expected_group_frames} 帧，"
            f"当前脚本固定为 {GROUP_FRAMES} 帧"
        )

    output.mkdir(parents=True, exist_ok=True)
    paths = {split: output / f"{split}.hdf5" for split in SPLITS}
    existing = [path for path in paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"输出已存在：{existing[0]}；确认重建时传 --overwrite")
    if args.overwrite:
        for path in existing:
            path.unlink()
        old_manifest = output / "manifest.json"
        if old_manifest.exists():
            old_manifest.unlink()

    attrs = {
        "format": FORMAT,
        "group_frames": GROUP_FRAMES,
        "group_duration_s": (GROUP_FRAMES - 1) / training_hz,
        "window_size": WINDOW_LENGTH,
        "windows_per_group": WINDOWS_PER_GROUP,
        "contact_groups_per_episode": CONTACT_GROUPS_PER_EPISODE,
        "noncontact_groups_per_episode": NONCONTACT_GROUPS_PER_EPISODE,
        "channels": len(ENCODER_INPUT_CHANNELS),
        "channel_names": json.dumps(list(ENCODER_INPUT_CHANNELS), ensure_ascii=False),
        "sample_hz": sample_hz,
        "training_hz": training_hz,
    }
    handles = {split: _create_output(paths[split], attrs) for split in SPLITS}
    audit = {split: Counter() for split in SPLITS}
    try:
        for number, path in enumerate(files, start=1):
            index = _episode_index(path)
            split = _split_for(index)
            rows, episode_audit = derive_episode(
                path,
                training_hz=training_hz,
                stride=stride,
            )
            _append(handles[split], rows, index)
            audit[split].update(episode_audit)
            if number % 10 == 0 or number == len(files):
                print(f"[{number}/{len(files)}]", flush=True)
    finally:
        for handle in handles.values():
            handle.close()

    split_summary = {split: _split_summary(paths[split]) for split in SPLITS}
    total = {
        key: sum(summary[key] for summary in split_summary.values())
        for key in next(iter(split_summary.values()))
        if key != "mask_all_true"
    }
    total["mask_all_true"] = all(
        summary["mask_all_true"] for summary in split_summary.values()
    )
    expected_groups = len(files) * (
        CONTACT_GROUPS_PER_EPISODE + NONCONTACT_GROUPS_PER_EPISODE
    )
    expected_windows = expected_groups * WINDOWS_PER_GROUP
    expected_contact_groups = len(files) * CONTACT_GROUPS_PER_EPISODE
    expected_noncontact_groups = len(files) * NONCONTACT_GROUPS_PER_EPISODE
    if (
        total["contact_groups"] != expected_contact_groups
        or total["noncontact_groups"] != expected_noncontact_groups
        or total["groups"] != expected_groups
        or total["windows"] != expected_windows
        or total["mask_all_true"] is not True
    ):
        raise RuntimeError(
            f"数据统计不符合预期：{total}；期望 "
            f"{expected_contact_groups}+{expected_noncontact_groups} 组，"
            f"{expected_windows} 窗口，mask 全为 True"
        )

    stats = compute_prepared_stats(paths["train"])
    (output / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    summary = {
        "format": FORMAT,
        "source": str(source.resolve()),
        "source_episodes": len(files),
        "group_frames": GROUP_FRAMES,
        "group_duration_s": (GROUP_FRAMES - 1) / training_hz,
        "window_length": WINDOW_LENGTH,
        "windows_per_group": WINDOWS_PER_GROUP,
        "splits": split_summary,
        "total": total,
        "audit": {split: dict(values) for split, values in audit.items()},
        "sample_kind": {
            str(SAMPLE_NONCONTACT): "noncontact",
            str(SAMPLE_PURE_CONTACT): "pure_contact",
            str(SAMPLE_MIXED_CONTACT): "mixed_contact",
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "README.md").write_text(FORMAT_README, encoding="utf-8")
    print(json.dumps(summary["total"], ensure_ascii=False, indent=2))
    print(f"完成：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

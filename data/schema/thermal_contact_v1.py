"""ThermalContact 原始 episode 契约。

触觉/thermal 以 100 Hz 采样，控制、机器人状态与 RGB 以 20 Hz 采样。
差分、重采样、动作块和归一化统计均由加载器派生，不落盘。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

FORMAT_VERSION = "thermalcontact-v2.0"
REGION_ORDER = ("lower", "middle", "upper", "handle")
COORDINATE_CONVENTION = (
    "right-handed; base=robot base frame; pose=[x,y,z,qw,qx,qy,qz]; "
    "quaternion=wxyz Hamilton; contact_force_link is in each finger link frame"
)

# 现有 thermal encoder 加载器使用的派生通道，不是落盘字段。
ENCODER_INPUT_CHANNELS = (
    "temp_left", "temp_right", "dtemp_left", "dtemp_right",
    "temp_rate_left", "temp_rate_right", "normal_force_left", "normal_force_right",
    "tangential_force_left", "tangential_force_right", "contact_area_left",
    "contact_area_right", "gripper_width", "contact_time",
)


@dataclass(frozen=True)
class FieldSpec:
    """HDF5 字段的尾形状、dtype 和含义；``None`` 表示配置尺寸。"""

    shape: tuple[int | None, ...]
    dtype: str
    description: str


SENSOR_FIELDS: dict[str, FieldSpec] = {
    "sensor/time": FieldSpec((), "f8", "100 Hz 传感器时间 (s)"),
    "tactile/fingertip_temp": FieldSpec((2,), "f4", "左/右指尖温度 (°C)"),
    "tactile/contact_force_link": FieldSpec((2, 3), "f4", "指尖 link 系三轴力 (N)"),
    "tactile/contact_normal_force": FieldSpec((2,), "f4", "左/右法向力 (N)"),
    "tactile/contact_area": FieldSpec((2,), "f4", "左/右接触面积 (m^2)"),
    "tactile/contact_mask": FieldSpec((2,), "?", "左/右物理接触标志"),
    "tactile/contact_event_id": FieldSpec((), "i4", "0=无接触，1..N=事件"),
    "tactile/sensor_position_base": FieldSpec((2, 3), "f4", "指垫中心在 base 系的位置 (m)"),
    "tactile/sensor_normal_base": FieldSpec((2, 3), "f4", "指垫内法向在 base 系的单位向量"),
    "privileged/link_temp": FieldSpec((len(REGION_ORDER),), "f4", "各热 link 真实温度 (°C)"),
    "privileged/contact_link_mask": FieldSpec((2, len(REGION_ORDER)), "?", "左/右指接触 link 归属"),
    "privileged/contact_valid": FieldSpec((2,), "?", "左/右指与目标杯子的有效接触"),
}

CONTROL_FIELDS: dict[str, FieldSpec] = {
    "control/time": FieldSpec((), "f8", "20 Hz 控制时间 (s)"),
    "control/sensor_index": FieldSpec((), "i8", "不晚于控制时刻的最新 sensor 索引"),
    "control/eef_target": FieldSpec((7,), "f4", "限速后实际送入 IK 的 TCP 绝对位姿"),
    "control/gripper_target": FieldSpec((), "f4", "夹爪宽度指令 (m)"),
    "control/target_force": FieldSpec((), "f4", "目标法向力 (N)"),
    "robot/eef_pose": FieldSpec((7,), "f4", "实测 TCP 位姿"),
    "robot/joint_position": FieldSpec((9,), "f4", "7 臂关节 (rad) + 2 指关节 (m)"),
    "robot/joint_velocity": FieldSpec((9,), "f4", "7 臂关节 (rad/s) + 2 指关节 (m/s)"),
    "robot/gripper_width": FieldSpec((), "f4", "夹爪宽度 (m)"),
    "camera/base_0_rgb": FieldSpec((None, None, 3), "u1", "固定相机 RGB"),
    "camera/wrist_0_rgb": FieldSpec((None, None, 3), "u1", "腕部相机 RGB"),
    "camera/wrist_0_pose": FieldSpec((7,), "f4", "腕部相机在 base 系的位姿"),
}

STATIC_FIELDS: dict[str, FieldSpec] = {
    "camera/base_0_intrinsics": FieldSpec((3, 3), "f4", "固定相机内参"),
    "camera/wrist_0_intrinsics": FieldSpec((3, 3), "f4", "腕部相机内参"),
    "camera/base_0_pose": FieldSpec((7,), "f4", "固定相机在 base 系的位姿"),
    "camera/hand_to_wrist_camera": FieldSpec((4, 4), "f4", "T_hand_camera"),
}

EVENT_FIELDS: dict[str, FieldSpec] = {
    "contact_events/start_sensor_index": FieldSpec((), "i8", "事件起始 sensor 索引"),
    "contact_events/end_sensor_index_exclusive": FieldSpec((), "i8", "事件右开结束 sensor 索引"),
}

FRAME_FIELDS = {**SENSOR_FIELDS, **CONTROL_FIELDS, **EVENT_FIELDS}
SCALAR_FIELDS = STATIC_FIELDS
EPISODE_METADATA_KEYS = (
    "episode_id", "random_seed", "split", "genesis_version", "coordinate_convention"
)


def empty_episode() -> dict[str, list]:
    return {name: [] for name in FRAME_FIELDS}


def _shape_matches(actual: tuple[int, ...], expected: tuple[int | None, ...]) -> bool:
    return len(actual) == len(expected) and all(e is None or a == e for a, e in zip(actual, expected))


def stack_episode(buffers: Mapping[str, list]) -> dict[str, np.ndarray]:
    """把多时间轴缓冲转为数组；事件列表允许为空。"""
    arrays: dict[str, np.ndarray] = {}
    for name, spec in FRAME_FIELDS.items():
        if name not in buffers:
            raise KeyError(f"缺少字段 {name}")
        values = buffers[name]
        if not values and name not in EVENT_FIELDS:
            raise ValueError(f"字段 {name} 没有任何样本")
        if values:
            arr = np.asarray(values, dtype=spec.dtype)
        else:
            arr = np.empty((0, *[int(v) for v in spec.shape]), dtype=spec.dtype)
        if not _shape_matches(arr.shape[1:], spec.shape):
            raise ValueError(f"字段 {name} 形状 {arr.shape}，尾形状应为 {spec.shape}")
        arrays[name] = arr
    return arrays


def _check_group_lengths(arrays: Mapping[str, np.ndarray], fields: Mapping[str, FieldSpec], label: str) -> list[str]:
    lengths = {name: int(np.asarray(arrays[name]).shape[0]) for name in fields if name in arrays}
    if not lengths:
        return []
    expected = next(iter(lengths.values()))
    return [f"{name} 长度 {n} 与 {label} 长度 {expected} 不一致" for name, n in lengths.items() if n != expected]


def validate_arrays(arrays: Mapping[str, np.ndarray], **_: Any) -> list[str]:
    """校验字段、形状、多时间轴和接触事件。"""
    issues: list[str] = []
    for name, spec in {**FRAME_FIELDS, **STATIC_FIELDS}.items():
        if name not in arrays:
            issues.append(f"缺少数据集 {name}")
            continue
        arr = np.asarray(arrays[name])
        tail = arr.shape if name in STATIC_FIELDS else arr.shape[1:]
        if not _shape_matches(tail, spec.shape):
            issues.append(f"{name} 形状 {arr.shape}与契约 {spec.shape} 不符")
    if issues:
        return issues

    issues += _check_group_lengths(arrays, SENSOR_FIELDS, "sensor")
    issues += _check_group_lengths(arrays, CONTROL_FIELDS, "control")
    issues += _check_group_lengths(arrays, EVENT_FIELDS, "contact_events")

    sensor_time = np.asarray(arrays["sensor/time"], dtype=np.float64)
    control_time = np.asarray(arrays["control/time"], dtype=np.float64)
    if sensor_time.size == 0 or control_time.size == 0:
        issues.append("sensor/control 时间轴不能为空")
        return issues
    if np.any(np.diff(sensor_time) <= 0.0):
        issues.append("sensor/time 不是严格递增")
    if np.any(np.diff(control_time) <= 0.0):
        issues.append("control/time 不是严格递增")

    sensor_index = np.asarray(arrays["control/sensor_index"], dtype=np.int64)
    if np.any(sensor_index < 0) or np.any(sensor_index >= sensor_time.size):
        issues.append("control/sensor_index 超出 sensor/time 范围")
    elif np.any(sensor_time[sensor_index] > control_time + 1e-9):
        issues.append("control/sensor_index 指向晚于当前控制时刻的传感器帧")

    normal = np.asarray(arrays["tactile/contact_normal_force"], dtype=np.float64)
    if not np.all(np.isfinite(normal)) or np.any(normal < -1e-6):
        issues.append("tactile/contact_normal_force 含非有限值或负值")
    directions = np.asarray(arrays["tactile/sensor_normal_base"], dtype=np.float64)
    if not np.allclose(np.linalg.norm(directions, axis=-1), 1.0, atol=1e-4):
        issues.append("tactile/sensor_normal_base 不是单位向量")

    starts = np.asarray(arrays["contact_events/start_sensor_index"], dtype=np.int64)
    ends = np.asarray(arrays["contact_events/end_sensor_index_exclusive"], dtype=np.int64)
    if np.any(starts < 0) or np.any(ends > sensor_time.size) or np.any(starts >= ends):
        issues.append("contact_events 存在越界或非正长度区间")
    if starts.size > 1 and np.any(starts[1:] < ends[:-1]):
        issues.append("contact_events 区间重叠或未排序")
    event_id = np.asarray(arrays["tactile/contact_event_id"], dtype=np.int64)
    if np.any(event_id < 0) or np.any(event_id > starts.size):
        issues.append("tactile/contact_event_id 超出 0..N")
    for i, (start, end) in enumerate(zip(starts, ends), start=1):
        if not np.all(event_id[start:end] == i):
            issues.append(f"接触事件 {i} 与 tactile/contact_event_id 不一致")
            break
    return issues


def metadata_for_spec(
    spec: Any, *, genesis_version: str, cup_pose: Any,
    cup_asset_version: str, outcome: Any,
) -> dict[str, Any]:
    """生成必需 attrs 和可审计扩展元数据。"""
    return {
        "episode_id": str(spec.episode_id),
        "random_seed": int(spec.random_seed),
        "split": str(spec.split),
        "genesis_version": str(genesis_version),
        "coordinate_convention": COORDINATE_CONVENTION,
        "dataset_version": str(spec.dataset_version),
        "region_order": list(spec.region_order),
        "target_region": int(spec.target_region),
        "language_instruction": str(getattr(spec, "instruction", "")),
        "contact_order": [int(v) for v in getattr(spec, "contact_order", ())],
        "contact_hold_s_by_region": [
            float(v) for v in getattr(spec, "contact_hold_s_by_region", ())
        ],
        "probe_positions": [
            [float(x) for x in value] for value in getattr(spec, "probe_positions", ())
        ],
        "probe_tool_dirs": [
            [float(x) for x in value] for value in getattr(spec, "probe_tool_dirs", ())
        ],
        "probe_finger_dirs": [
            [float(x) for x in value] for value in getattr(spec, "probe_finger_dirs", ())
        ],
        "cup_pose": [float(v) for v in np.asarray(cup_pose).reshape(-1)],
        "cup_asset_version": str(cup_asset_version),
        "material_parameters": {k: float(v) for k, v in spec.material_parameters},
        "sensor_RC": float(spec.sensor_rc_s),
        "sensor_noise": float(spec.sensor_noise_c),
        "sensor_bias": float(spec.sensor_bias_c),
        "quantization": float(spec.sensor_quantization_c),
        "target_force_n": float(spec.target_force_n),
        "initial_finger_temperature": [float(v) for v in spec.fingertip_temps_c],
        "initial_link_temperature": [float(v) for v in spec.link_temps_c],
        "contact_quality": str(spec.contact_quality),
        "termination_reason": str(outcome.termination_reason),
        "policy_steps": int(outcome.policy_steps),
        "peak_normal_force_n": float(outcome.peak_normal_force_n),
    }


__all__ = [
    "COORDINATE_CONVENTION", "CONTROL_FIELDS", "ENCODER_INPUT_CHANNELS",
    "EPISODE_METADATA_KEYS", "EVENT_FIELDS", "FORMAT_VERSION", "FRAME_FIELDS",
    "FieldSpec", "REGION_ORDER", "SCALAR_FIELDS", "SENSOR_FIELDS", "STATIC_FIELDS",
    "empty_episode", "metadata_for_spec", "stack_episode", "validate_arrays",
]

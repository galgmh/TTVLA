"""ThermalContact-v1 原始数据契约（HDF5 ``thermalcontact-v1.8``）。

一份文件 = 一集 = 一次单区域受控触碰。文件里只存**原始量**：

* ``obs/`` 是策略在仿真里能读到的传感器序列（指尖温度、指尖接触力、
  夹爪开口、指位、TCP 位姿、凝胶标记估计的接触面积）；
* ``action/`` 是控制器实际下发的指令（TCP 位姿目标、夹爪开口目标、
  目标法向力、目标区域、阶段码）；
* ``privileged/`` 是仿真真值（四段 link 温度、手指-热区接触归属、是否
  纯接触），**只用于训练标签与审计，绝不进入模型输入**。

派生量（ΔT、温度斜率、前缀窗口、归一化统计）一律由加载器现算，
不落盘——原始序列是唯一真源。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

# 数据格式版本：与任务版本 ThermalContact-v1 绑定，字段变动必须升版本号。
# v1.8：删除保留的 BASELINE 阶段码（阶段码整体前移一位），
#       并删除任务阈值 τ（模型只保留温度回归与接触有效性两个头）。
FORMAT_VERSION = "thermalcontact-v1.8"

# 区域顺序（= 资产清单 region_order = privileged/link_temp 的列顺序）。
REGION_ORDER = ("lower", "middle", "upper", "handle")

# 阶段码：整数落盘，名字只在代码与文档里用。
PHASES = (
    "RESET",
    "APPROACH",
    "SEEK_CONTACT",
    "HOLD",
    "RETRACT",
    "DONE",
    "FAILED",
)
PHASE_CODES = {name: i for i, name in enumerate(PHASES)}
PHASE_NAMES = {i: name for name, i in PHASE_CODES.items()}

# 每个训练样本的输入通道（逐时刻 14 维；夹爪开口与接触时间是全局量）。
ENCODER_INPUT_CHANNELS = (
    "temp_left",
    "temp_right",
    "dtemp_left",
    "dtemp_right",
    "temp_rate_left",
    "temp_rate_right",
    "normal_force_left",
    "normal_force_right",
    "tangential_force_left",
    "tangential_force_right",
    "contact_area_left",
    "contact_area_right",
    "gripper_width",
    "contact_time",
)


@dataclass(frozen=True)
class FieldSpec:
    """一个 HDF5 数据集的形状尾、dtype 与含义。"""

    shape: tuple[int, ...]
    dtype: str
    description: str


# 逐帧数据集：形状尾不含帧数 L。
FRAME_FIELDS: dict[str, FieldSpec] = {
    "time/sim_time": FieldSpec((), "f8", "仿真时间 (s)"),
    "obs/fingertip_temp": FieldSpec((2,), "f8", "指尖温度 (左, 右) (°C)"),
    "obs/contact_force_world": FieldSpec((2, 3), "f8", "指尖接触力，世界系 (N)"),
    "obs/contact_normal_force": FieldSpec((2,), "f8", "指尖法向力 (N)"),
    "obs/contact_mask": FieldSpec((2,), "?", "指尖接触判定"),
    "obs/gripper_width": FieldSpec((), "f8", "夹爪开口 (m)"),
    "obs/finger_joint_position": FieldSpec((2,), "f8", "指关节位置 (m)"),
    "obs/eef_pose": FieldSpec((7,), "f8", "TCP 位姿 (pos 3 + quat 4, wxyz)"),
    "obs/contact_area": FieldSpec((2,), "f8", "凝胶标记估计的接触面积 (m²)"),
    "action/eef_target": FieldSpec((7,), "f8", "TCP 指令位姿 (pos 3 + quat 4)"),
    "action/gripper_target": FieldSpec((), "f8", "夹爪开口指令 (m)"),
    "action/target_force": FieldSpec((), "f8", "目标法向力 (N)"),
    "action/probe_region": FieldSpec((), "i8", "目标区域编号；混杂接触为 -1"),
    "action/phase": FieldSpec((), "i8", "阶段码（见 PHASE_CODES）"),
    "privileged/link_temp": FieldSpec((4,), "f8", "四段 link 温度 (°C)"),
    "privileged/contact_link_mask": FieldSpec((2, 4), "?", "手指-热区接触归属"),
    "privileged/pure_contact": FieldSpec((), "?", "本帧是否为纯目标区域接触"),
    "privileged/contact_valid": FieldSpec((), "?", "本帧是否有杯子接触"),
}

# 标量数据集（非序列）。
SCALAR_FIELDS: dict[str, FieldSpec] = {
    "privileged/target_region": FieldSpec((), "i8", "本集目标区域；混杂为 -1"),
}

# 必需的 episode 元数据键（写成文件属性里的 JSON）。
EPISODE_METADATA_KEYS = (
    "episode_id",
    "dataset_version",
    "random_seed",
    "split",
    "region_order",
    "target_region",
    "regime",
    "finger_mode",
    "cup_pose",
    "cup_asset_version",
    "material_parameters",
    "sensor_RC",
    "sensor_noise",
    "sensor_bias",
    "quantization",
    "target_force_level",
    "target_force_n",
    "initial_finger_temperature",
    "link_temperature",
    "contact_quality",
    "contact_hold_s",
    "contact_start_time",
    "contact_end_time",
    "termination_reason",
    "policy_steps",
    "peak_normal_force_n",
)


def empty_episode() -> dict[str, list]:
    """给记录器用的空缓冲：逐帧数据集各一个 list。"""
    return {name: [] for name in FRAME_FIELDS}


def stack_episode(buffers: Mapping[str, list]) -> dict[str, np.ndarray]:
    """把逐帧缓冲整理成 ``{字段: 数组}``（形状 (L, *尾)）。"""
    arrays: dict[str, np.ndarray] = {}
    for name, spec in FRAME_FIELDS.items():
        if name not in buffers:
            raise KeyError(f"缺少字段 {name}")
        values = buffers[name]
        if not values:
            raise ValueError(f"字段 {name} 没有任何帧")
        arr = np.asarray(values, dtype=spec.dtype)
        expected = (len(values), *spec.shape)
        if arr.shape != expected:
            raise ValueError(f"字段 {name} 形状 {arr.shape}，应为 {expected}")
        arrays[name] = arr
    return arrays


def validate_arrays(arrays: Mapping[str, np.ndarray], *, atol_time: float = 1e-9) -> list[str]:
    """校验一份 episode 的数组：形状、时间单调、取值域。返回问题列表。

    只查契约层面的硬约束；物理合理性（温度曲线是否连续、接触是否纯）
    由 ``data.validation`` 的检查器负责。
    """
    issues: list[str] = []
    for name, spec in FRAME_FIELDS.items():
        if name not in arrays:
            issues.append(f"缺少数据集 {name}")
            continue
        arr = np.asarray(arrays[name])
        if arr.ndim != 1 + len(spec.shape):
            issues.append(f"{name} 维数 {arr.ndim}，应为 {1 + len(spec.shape)}")
            continue
        if spec.shape and arr.shape[1:] != spec.shape:
            issues.append(f"{name} 形状尾 {arr.shape[1:]}，应为 {spec.shape}")
        if arr.shape[0] == 0:
            issues.append(f"{name} 长度为 0")
    if issues:
        return issues

    n = len(arrays["time/sim_time"])
    for name, arr in arrays.items():
        if arr.shape[0] != n:
            issues.append(f"{name} 长度 {arr.shape[0]} 与帧数 {n} 不一致")

    t = np.asarray(arrays["time/sim_time"], dtype=np.float64)
    if n > 1 and np.any(np.diff(t) <= 0.0):
        issues.append("time/sim_time 不是严格递增")

    phase = np.asarray(arrays["action/phase"])
    bad = np.setdiff1d(np.unique(phase), np.arange(len(PHASES)))
    if bad.size:
        issues.append(f"action/phase 出现未知阶段码 {bad.tolist()}")

    temp = np.asarray(arrays["obs/fingertip_temp"], dtype=np.float64)
    if not np.all(np.isfinite(temp)):
        issues.append("obs/fingertip_temp 含非有限值")
    elif np.nanmin(temp) < -50.0 or np.nanmax(temp) > 200.0:
        issues.append(f"obs/fingertip_temp 超出合理范围 [{np.nanmin(temp):.1f}, {np.nanmax(temp):.1f}] °C")

    force = np.asarray(arrays["obs/contact_normal_force"], dtype=np.float64)
    if np.nanmin(force) < -1e-6:
        issues.append("obs/contact_normal_force 出现负值")
    if np.nanmax(force) > 50.0:
        issues.append(f"obs/contact_normal_force 异常大（{np.nanmax(force):.1f} N），可能撞穿杯子")

    return issues


def metadata_for_spec(spec: Any, *, cup_pose, cup_asset_version: str, outcome: Any) -> dict[str, Any]:
    """从 ``EpisodeSpec`` 与控制器结局组装 episode 元数据（JSON 可序列化）。"""
    return {
        "episode_id": spec.episode_id,
        "dataset_version": spec.dataset_version,
        "random_seed": int(spec.random_seed),
        "split": spec.split,
        "region_order": list(spec.region_order),
        "target_region": int(spec.target_region),
        "regime": spec.regime,
        "finger_mode": spec.finger_mode,
        "cup_pose": [float(v) for v in np.asarray(cup_pose, dtype=np.float64).reshape(-1)],
        "cup_asset_version": cup_asset_version,
        "material_parameters": {k: float(v) for k, v in spec.material_parameters},
        "sensor_RC": float(spec.sensor_rc_s),
        "sensor_noise": float(spec.sensor_noise_c),
        "sensor_bias": float(spec.sensor_bias_c),
        "quantization": float(spec.sensor_quantization_c),
        "target_force_level": float(spec.force_level),
        "target_force_n": float(spec.target_force_n),
        "initial_finger_temperature": [float(v) for v in spec.fingertip_temps_c],
        "link_temperature": [float(v) for v in spec.link_temps_c],
        "contact_quality": spec.contact_quality,
        "contact_hold_s": float(spec.contact_hold_s),
        "contact_start_time": outcome.contact_start_s,
        "contact_end_time": outcome.contact_end_s,
        "termination_reason": outcome.termination_reason,
        "policy_steps": int(outcome.policy_steps),
        "peak_normal_force_n": float(outcome.peak_normal_force_n),
    }


__all__ = [
    "ENCODER_INPUT_CHANNELS",
    "EPISODE_METADATA_KEYS",
    "FORMAT_VERSION",
    "FRAME_FIELDS",
    "FieldSpec",
    "PHASES",
    "PHASE_CODES",
    "PHASE_NAMES",
    "REGION_ORDER",
    "SCALAR_FIELDS",
    "empty_episode",
    "metadata_for_spec",
    "stack_episode",
    "validate_arrays",
]

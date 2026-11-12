"""单集规格（EpisodeSpec）：采样结果的不可变快照。

一集 = 一次单区域受控触碰。规格在采样时一次性算好，之后只读：场景复位、
控制器、记录器都从同一份规格取值，任何模块都不再自己抽随机数，因此同一
(random_seed, split, index) 必然复现同一集。

规格里的字段分三类：

* 仿真输入 —— 杯子位姿抖动、四段 link 温度、指尖初始温度、传感器参数、
  探测点（作者坐标系下的位置与接近方向）、目标力与接触质量；
* 标签口径 —— 目标区域；
* 记录用元数据 —— 数据集版本、划分、随机种子等。

模型输入只来自传感器序列；``target_region`` / ``link_temps_c`` / τ 等
都是特权信息，绝不进入编码器。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class EpisodeSpec:
    """一集的全部输入与元数据（不可变）。"""

    # ---- 身份与追溯
    episode_id: str
    dataset_version: str
    split: str
    random_seed: int
    # 区域顺序（= 资产清单 region_order，也是 link_temp 的列顺序）。
    region_order: tuple[str, ...]

    # ---- 采样标签
    target_region: int  # 0..3；混杂接触集为 -1
    mixed: bool
    regime: str  # case_1 … case_5（四段温度情景）
    finger_mode: str  # ambient / history

    # ---- 仿真输入：温度
    link_temps_c: tuple[float, ...]  # 按 region_order 的四段温度
    fingertip_temps_c: tuple[float, float]  # (左, 右)

    # ---- 仿真输入：力与接触质量
    force_level: float
    target_force_n: float
    contact_quality: str  # stable / brief / failed
    contact_hold_s: float  # 稳定接触的保持时长；failed 为 0
    fail_retreat_m: float  # 仅 failed 非零：停在接触点前方这么远

    # ---- 仿真输入：传感器非理想性
    sensor_rc_s: float
    sensor_noise_c: float
    sensor_bias_c: float
    sensor_quantization_c: float

    # ---- 仿真输入：探测点（杯子作者坐标系）
    probe_pos: tuple[float, float, float]
    probe_tool_dir: tuple[float, float, float]
    probe_finger_dir: tuple[float, float, float]
    probe_pre_dist: float

    # ---- 仿真输入：杯子摆放抖动（相对槽位）
    cup_xy_jitter_m: tuple[float, float]
    cup_yaw_jitter_rad: float

    # ---- 记录用：材质参数（有序元组，保证可哈希/不可变）
    material_parameters: tuple[tuple[str, float], ...] = field(default_factory=tuple)

    # ---- 多接触 episode（空元组表示旧的单接触规格）
    instruction: str = ""
    contact_order: tuple[int, ...] = field(default_factory=tuple)
    contact_hold_s_by_region: tuple[float, ...] = field(default_factory=tuple)
    probe_positions: tuple[tuple[float, float, float], ...] = field(default_factory=tuple)
    probe_tool_dirs: tuple[tuple[float, float, float], ...] = field(default_factory=tuple)
    probe_finger_dirs: tuple[tuple[float, float, float], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if len(self.region_order) != 4:
            raise ValueError(f"region_order 必须是 4 项，实为 {self.region_order}")
        if len(self.link_temps_c) != 4:
            raise ValueError(f"link_temps_c 必须是 4 项，实为 {self.link_temps_c}")
        if not -2 <= int(self.target_region) < 4:
            raise ValueError(f"target_region 必须在 [-2, 3]，实为 {self.target_region}")
        if int(self.target_region) == -1 and not self.mixed:
            raise ValueError("target_region == -1 专用于 mixed contact")
        if int(self.target_region) == -2 and self.mixed:
            raise ValueError("target_region == -2 表示多区域顺序接触，不是 mixed contact")
        if self.contact_quality not in ("stable", "brief", "failed"):
            raise ValueError(f"未知接触质量 {self.contact_quality!r}")
        if self.contact_hold_s < 0.0:
            raise ValueError(f"contact_hold_s 必须 >= 0，实为 {self.contact_hold_s}")
        if self.target_force_n <= 0.0:
            raise ValueError(f"target_force_n 必须为正，实为 {self.target_force_n}")
        if not 0.0 <= self.sensor_rc_s < 5.0:
            raise ValueError(f"sensor_rc_s 超出合理范围：{self.sensor_rc_s}")
        if self.sensor_quantization_c < 0.0 or self.sensor_noise_c < 0.0:
            raise ValueError("噪声与量化必须非负")
        if self.contact_order:
            if sorted(self.contact_order) != list(range(4)):
                raise ValueError(f"contact_order 必须是 0..3 的全排列：{self.contact_order}")
            for name, values in (
                ("contact_hold_s_by_region", self.contact_hold_s_by_region),
                ("probe_positions", self.probe_positions),
                ("probe_tool_dirs", self.probe_tool_dirs),
                ("probe_finger_dirs", self.probe_finger_dirs),
            ):
                if len(values) != 4:
                    raise ValueError(f"{name} 必须是按 region_order 排列的 4 项")

    # ------------------------------------------------------------ 便捷派生值
    @property
    def target_region_name(self) -> str:
        if self.target_region == -2:
            return "multi"
        return "mixed" if self.target_region < 0 else self.region_order[int(self.target_region)]

    @property
    def material_parameters_dict(self) -> dict[str, float]:
        return dict(self.material_parameters)

    def to_dict(self) -> dict[str, Any]:
        """JSON 可序列化的记录字典。"""
        return {
            "episode_id": self.episode_id,
            "dataset_version": self.dataset_version,
            "split": self.split,
            "random_seed": int(self.random_seed),
            "region_order": list(self.region_order),
            "target_region": int(self.target_region),
            "mixed": bool(self.mixed),
            "regime": self.regime,
            "finger_mode": self.finger_mode,
            "link_temps_c": [float(v) for v in self.link_temps_c],
            "fingertip_temps_c": [float(v) for v in self.fingertip_temps_c],
            "force_level": float(self.force_level),
            "target_force_n": float(self.target_force_n),
            "contact_quality": self.contact_quality,
            "contact_hold_s": float(self.contact_hold_s),
            "fail_retreat_m": float(self.fail_retreat_m),
            "sensor_rc_s": float(self.sensor_rc_s),
            "sensor_noise_c": float(self.sensor_noise_c),
            "sensor_bias_c": float(self.sensor_bias_c),
            "sensor_quantization_c": float(self.sensor_quantization_c),
            "probe_pos": [float(v) for v in self.probe_pos],
            "probe_tool_dir": [float(v) for v in self.probe_tool_dir],
            "probe_finger_dir": [float(v) for v in self.probe_finger_dir],
            "probe_pre_dist": float(self.probe_pre_dist),
            "cup_xy_jitter_m": [float(v) for v in self.cup_xy_jitter_m],
            "cup_yaw_jitter_rad": float(self.cup_yaw_jitter_rad),
            "material_parameters": {k: float(v) for k, v in self.material_parameters},
            "instruction": self.instruction,
            "contact_order": [int(v) for v in self.contact_order],
            "contact_hold_s_by_region": [float(v) for v in self.contact_hold_s_by_region],
            "probe_positions": [[float(x) for x in v] for v in self.probe_positions],
            "probe_tool_dirs": [[float(x) for x in v] for v in self.probe_tool_dirs],
            "probe_finger_dirs": [[float(x) for x in v] for v in self.probe_finger_dirs],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EpisodeSpec":
        """从记录字典还原（校验交给 ``__post_init__``）。"""
        return cls(
            episode_id=str(data["episode_id"]),
            dataset_version=str(data["dataset_version"]),
            split=str(data["split"]),
            random_seed=int(data["random_seed"]),
            region_order=tuple(str(v) for v in data["region_order"]),
            target_region=int(data["target_region"]),
            mixed=bool(data["mixed"]),
            regime=str(data["regime"]),
            finger_mode=str(data["finger_mode"]),
            link_temps_c=tuple(float(v) for v in data["link_temps_c"]),
            fingertip_temps_c=tuple(float(v) for v in data["fingertip_temps_c"]),
            force_level=float(data["force_level"]),
            target_force_n=float(data["target_force_n"]),
            contact_quality=str(data["contact_quality"]),
            contact_hold_s=float(data["contact_hold_s"]),
            fail_retreat_m=float(data["fail_retreat_m"]),
            sensor_rc_s=float(data["sensor_rc_s"]),
            sensor_noise_c=float(data["sensor_noise_c"]),
            sensor_bias_c=float(data["sensor_bias_c"]),
            sensor_quantization_c=float(data["sensor_quantization_c"]),
            probe_pos=tuple(float(v) for v in data["probe_pos"]),
            probe_tool_dir=tuple(float(v) for v in data["probe_tool_dir"]),
            probe_finger_dir=tuple(float(v) for v in data["probe_finger_dir"]),
            probe_pre_dist=float(data["probe_pre_dist"]),
            cup_xy_jitter_m=tuple(float(v) for v in data["cup_xy_jitter_m"]),
            cup_yaw_jitter_rad=float(data["cup_yaw_jitter_rad"]),
            material_parameters=tuple(
                sorted((str(k), float(v)) for k, v in data["material_parameters"].items())
            ),
            instruction=str(data.get("instruction", "")),
            contact_order=tuple(int(v) for v in data.get("contact_order", ())),
            contact_hold_s_by_region=tuple(float(v) for v in data.get("contact_hold_s_by_region", ())),
            probe_positions=tuple(tuple(float(x) for x in v) for v in data.get("probe_positions", ())),
            probe_tool_dirs=tuple(tuple(float(x) for x in v) for v in data.get("probe_tool_dirs", ())),
            probe_finger_dirs=tuple(tuple(float(x) for x in v) for v in data.get("probe_finger_dirs", ())),
        )


__all__ = ["EpisodeSpec"]

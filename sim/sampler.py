"""确定性采样器：给定 (数据集种子, 划分, 序号) 生成一集规格。

采样规则（``sim/generation/presets/*.yaml``）只在这里落地，其他模块不抽随机数：

* 目标区域：试生产按 4 区域均衡轮转，正式划分按均匀分布；
* 四段 link 初始温度：从 ``region_temps_c`` 的 5 种情景里均匀抽一种，
  每段在情景给定的固定区间内均匀采样；
* 指尖初始温度 70% 直接初始化在常温（传感器配置的 ``ambient_temp_c``）、
  30% 带历史余温；左右差默认很小，只有 ``test_finger_history`` 划分刻意
  放大到 4–10 K；
* 力档位、接触质量、传感器非理想性逐集抽样；``test_sensor_shift`` 用
  更宽的范围；
* 探测点 = 任务配置里该区域的位置 + 小幅抖动（不跨热区分界）；
* 杯子位姿抖动相对槽位。

确定性来自 ``numpy.random.SeedSequence([seed, crc32(split), index])``：
与遍历顺序无关，任意子集可单独重放。
"""

from __future__ import annotations

import itertools
import zlib
from typing import Any

import numpy as np

from sim.episode import EpisodeSpec

SPLITS = (
    "train",
    "validation",
    "test_id",
    "test_finger_history",
    "test_sensor_shift",
    "test_mixed_contact",
    "pilot",
)


def _split_code(split: str) -> int:
    """稳定的划分编号（不用 Python 的 hash，它按进程加盐）。"""
    return int(zlib.crc32(split.encode("utf-8")))


def _categorical(rng: np.random.Generator, probs: dict[str, float]) -> str:
    """按概率抽一个类别名。"""
    keys = list(probs)
    weights = np.asarray([float(probs[k]) for k in keys], dtype=np.float64)
    if weights.sum() <= 0.0:
        raise ValueError(f"概率之和必须为正：{probs}")
    return str(keys[int(rng.choice(len(keys), p=weights / weights.sum()))])


class ThermalContactSampler:
    """按数据集配置采样 ``EpisodeSpec``。"""

    def __init__(
        self,
        task_cfg: dict[str, Any],
        dataset_cfg: dict[str, Any],
        split: str,
        *,
        seed: int | None = None,
        count: int | None = None,
    ):
        if split not in SPLITS:
            raise ValueError(f"未知划分 {split!r}；可用：{list(SPLITS)}")
        self.task = task_cfg
        self.ds = dataset_cfg
        self.split = split
        self.seed = int(dataset_cfg["seed"] if seed is None else seed)
        if count is None:
            # 在线环境只需可复现地采一集；离线生成器必须显式传入数量。
            count = 1
        if int(count) < 1:
            raise ValueError(f"count 必须为正整数，实为 {count}")
        self.count = int(count)
        self.region_order = tuple(str(v) for v in task_cfg["object"]["region_order"])
        # 四段 link 的初始温度情景：每种情景给每个区域一个固定区间。
        self.region_temps: list[dict[str, tuple[float, float]]] = []
        for i, case in enumerate(dataset_cfg["region_temps_c"]):
            if set(case) != set(self.region_order):
                raise ValueError(
                    f"region_temps_c[{i}] 的区域 {sorted(case)} 必须与 region_order "
                    f"{sorted(self.region_order)} 一致"
                )
            self.region_temps.append(
                {name: (float(case[name][0]), float(case[name][1])) for name in self.region_order}
            )
        self._material = dict(task_cfg["object"]["material_properties"][task_cfg["object"]["material"]])
        self.multi_contact = bool(dataset_cfg.get("multi_contact", False))
        self.instruction = str(
            task_cfg.get("instruction")
            or task_cfg.get("instructions", [""])[0]
        )
        self.contact_orders = tuple(itertools.permutations(range(4)))
        self.duration_schedules = tuple(
            tuple(float(v) for v in row)
            for row in dataset_cfg.get("contact_duration_schedules", ())
        )
        if self.multi_contact:
            expected = len(self.region_temps) * len(self.contact_orders) * len(self.duration_schedules)
            if self.count > expected:
                raise ValueError(
                    f"多接触析因空间最多为 {len(self.region_temps)}×{len(self.contact_orders)}×"
                    f"{len(self.duration_schedules)}={expected} 集，请求为 {self.count}"
                )
        # 区域 z 范围：校验探测点抖动不越界。
        self._z_spans = {k: tuple(float(v) for v in span) for k, span in task_cfg["object"]["region_z_spans"].items()}

    # ------------------------------------------------------------------ 采样
    def sample(self, index: int) -> EpisodeSpec:
        """第 ``index`` 集的规格。"""
        if not 0 <= index < self.count:
            raise IndexError(f"index {index} 超出划分 {self.split} 的 {self.count} 集")
        seq = np.random.SeedSequence([self.seed, _split_code(self.split), int(index)])
        rng = np.random.default_rng(seq)
        random_seed = int(seq.generate_state(1)[0])
        if self.multi_contact:
            return self._sample_multi(index, rng, random_seed)
        mixed = self.split == "test_mixed_contact"

        # ---- 目标区域与任务阈值
        if mixed:
            target_region = -1
        elif self.split == "pilot":
            # 试生产：4 区域 x 250，严格均衡。
            target_region = int(index % 4)
        else:
            target_region = int(rng.integers(0, 4))
        # ---- 四段 link 的初始温度：先选一种情景，再逐段在固定区间内采样
        case_index = int(rng.integers(0, len(self.region_temps)))
        case = self.region_temps[case_index]
        regime = f"case_{case_index + 1}"
        temps = tuple(float(rng.uniform(*case[name])) for name in self.region_order)

        # ---- 指尖初始温度
        finger_cfg = self.ds["fingertip"]
        if self.split == "test_finger_history":
            finger_mode = "history"
        else:
            finger_mode = "history" if rng.random() < float(finger_cfg["history_probability"]) else "ambient"
        if finger_mode == "ambient":
            # 常温：左右指尖都直接初始化在室温（= 传感器配置的 ambient_temp_c），
            # 记录从第一帧起就是稳定值。
            ambient = float(self.task["sensor"]["ambient_temp_c"])
            fingertip = tuple(
                float(v) for v in np.clip([ambient, ambient], *finger_cfg["physical_range_c"])
            )
        else:
            base = float(rng.uniform(*finger_cfg["history_mode_c"]))
            if self.split == "test_finger_history":
                # 刻意放大左右差：编码器必须能分别处理两根手指的历史。
                asymmetry = float(rng.uniform(4.0, 10.0)) * (1.0 if rng.random() < 0.5 else -1.0)
                left, right = base + 0.5 * asymmetry, base - 0.5 * asymmetry
            else:
                delta = float(rng.uniform(*finger_cfg["pair_delta_c"]))
                left, right = base + delta, base - delta
            fingertip = tuple(
                float(v) for v in np.clip([left, right], *finger_cfg["physical_range_c"])
            )

        # ---- 接触质量；夹爪宽度目标统一为 0，不采样目标力档位。
        contact_quality = _categorical(rng, self.ds["contact_quality_probabilities"])
        contact_cfg = self.task["contact"]
        if contact_quality == "stable":
            hold = float(self.task["timing"]["contact_s"])
        elif contact_quality == "brief":
            hold = float(rng.uniform(*contact_cfg["brief_hold_s"]))
        else:
            hold = 0.0

        # ---- 传感器非理想性
        sensor_cfg = self.ds["sensor_randomization"]
        nominal = self.task["sensor"]["nominal"]
        shift = self.split == "test_sensor_shift"
        if shift:
            ranges = sensor_cfg["shift_test"]
            rc_mult = rng.uniform(*ranges["rc_multiplier_range"])
            noise_mult = rng.uniform(*ranges["noise_multiplier_range"])
            quant_mult = rng.uniform(*ranges["quantization_multiplier_range"])
            bias = float(rng.uniform(*ranges["bias_absolute_range"]))
        else:
            rc_mult = rng.uniform(*sensor_cfg["rc_multiplier_range"])
            noise_mult = rng.uniform(*sensor_cfg["noise_multiplier_range"])
            quant_mult = rng.uniform(*sensor_cfg["quantization_multiplier_range"])
            bias = float(rng.uniform(*sensor_cfg["bias_absolute_range"]))
        rc = float(nominal["rc_s"]) * float(rc_mult)
        noise = float(nominal["noise_c"]) * float(noise_mult)
        quantization = float(nominal["quantization_c"]) * float(quant_mult)

        # ---- 探测点（作者坐标系）+ 抖动
        base_probe = self.task["mixed_probe"] if mixed else self.task["regions"][self.region_order[target_region]]
        xyz = np.asarray(base_probe["pos"], dtype=np.float64).copy()
        xy_j = float(base_probe["xy_jitter_m"])
        z_j = float(base_probe["z_jitter_m"])
        xyz[0] += float(rng.uniform(-xy_j, xy_j))
        xyz[1] += float(rng.uniform(-xy_j, xy_j))
        xyz[2] += float(rng.uniform(-z_j, z_j))
        if not mixed:
            lo, hi = self._z_spans[self.region_order[target_region]]
            if not (lo <= xyz[2] <= hi):
                raise ValueError(
                    f"探测点 z={xyz[2]:.4f} 超出区域 {self.region_order[target_region]} 的 [{lo}, {hi}]"
                )

        # ---- 杯子位姿抖动
        object_cfg = self.task["object"]
        jitter_xy = float(object_cfg["xy_jitter_m"])
        cup_xy = (float(rng.uniform(-jitter_xy, jitter_xy)), float(rng.uniform(-jitter_xy, jitter_xy)))
        cup_yaw = float(rng.uniform(-float(object_cfg["yaw_jitter_rad"]), float(object_cfg["yaw_jitter_rad"])))

        return EpisodeSpec(
            episode_id=f"{self.ds['dataset_version']}_{self.split}_{index:06d}",
            dataset_version=str(self.ds["dataset_version"]),
            split=self.split,
            random_seed=random_seed,
            region_order=self.region_order,
            target_region=target_region,
            mixed=mixed,
            regime=regime,
            finger_mode=finger_mode,
            link_temps_c=tuple(float(v) for v in temps),
            fingertip_temps_c=fingertip,
            contact_quality=contact_quality,
            contact_hold_s=hold,
            sensor_rc_s=rc,
            sensor_noise_c=noise,
            sensor_bias_c=bias,
            sensor_quantization_c=quantization,
            probe_pos=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
            probe_tool_dir=tuple(float(v) for v in base_probe["tool_dir"]),
            probe_finger_dir=tuple(float(v) for v in base_probe["finger_dir"]),
            probe_pre_dist=float(base_probe["pre_dist"]),
            cup_xy_jitter_m=cup_xy,
            cup_yaw_jitter_rad=cup_yaw,
            material_parameters=tuple(sorted((str(k), float(v)) for k, v in self._material.items())),
        )

    @staticmethod
    def _rotate_about_axis(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
        """Rodrigues 旋转，用于在不破坏 tool/finger 正交性的前提下加倾斜。"""
        axis = np.asarray(axis, dtype=np.float64)
        axis = axis / np.linalg.norm(axis)
        vector = np.asarray(vector, dtype=np.float64)
        return (
            vector * np.cos(angle)
            + np.cross(axis, vector) * np.sin(angle)
            + axis * float(axis @ vector) * (1.0 - np.cos(angle))
        )

    def _sample_probe(self, rng: np.random.Generator, region_index: int):
        """为一个物理区域采样高度、绕杯方位角和水平倾斜。"""
        name = self.region_order[region_index]
        cfg = self.task["regions"][name]
        xyz = np.asarray(cfg["pos"], dtype=np.float64).copy()
        xy_jitter = float(cfg.get("xy_jitter_m", 0.0))
        xyz[:2] += rng.uniform(-xy_jitter, xy_jitter, size=2)
        xyz[2] += float(rng.uniform(-float(cfg.get("z_jitter_m", 0.0)), float(cfg.get("z_jitter_m", 0.0))))

        azimuth = 0.0 if name == "handle" else float(
            rng.uniform(-float(cfg.get("azimuth_jitter_rad", 0.0)), float(cfg.get("azimuth_jitter_rad", 0.0)))
        )
        c, s = np.cos(azimuth), np.sin(azimuth)
        rz = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        tool = rz @ np.asarray(cfg["tool_dir"], dtype=np.float64)
        finger = rz @ np.asarray(cfg["finger_dir"], dtype=np.float64)
        tilt = float(rng.uniform(-float(cfg.get("tilt_jitter_rad", 0.0)), float(cfg.get("tilt_jitter_rad", 0.0))))
        tool = self._rotate_about_axis(tool, finger, tilt)
        tool /= np.linalg.norm(tool)
        finger -= float(finger @ tool) * tool
        finger /= np.linalg.norm(finger)

        lo, hi = self._z_spans[name]
        if not lo <= xyz[2] <= hi:
            raise ValueError(f"探测点 z={xyz[2]:.4f} 超出区域 {name} 的 [{lo}, {hi}]")
        return tuple(xyz), tuple(tool), tuple(finger)

    def _sample_multi(
        self, index: int, rng: np.random.Generator, random_seed: int
    ) -> EpisodeSpec:
        """5 温度情景 × 24 全排列 × 12 平衡时长方案的确定性展开。"""
        per_case = len(self.contact_orders) * len(self.duration_schedules)
        case_index, remainder = divmod(index, per_case)
        order_index, schedule_index = divmod(remainder, len(self.duration_schedules))
        case = self.region_temps[case_index]
        temps = tuple(float(rng.uniform(*case[name])) for name in self.region_order)

        ambient = float(self.task["sensor"]["ambient_temp_c"])
        fingertip = (ambient, ambient)

        sensor_cfg = self.ds["sensor_randomization"]
        nominal = self.task["sensor"]["nominal"]
        rc = float(nominal["rc_s"]) * float(rng.uniform(*sensor_cfg["rc_multiplier_range"]))
        noise = float(nominal["noise_c"]) * float(rng.uniform(*sensor_cfg["noise_multiplier_range"]))
        quant = float(nominal["quantization_c"]) * float(
            rng.uniform(*sensor_cfg["quantization_multiplier_range"])
        )
        bias = float(rng.uniform(*sensor_cfg["bias_absolute_range"]))

        probes = [self._sample_probe(rng, i) for i in range(4)]
        object_cfg = self.task["object"]
        jitter_xy = float(object_cfg["xy_jitter_m"])
        cup_xy = (float(rng.uniform(-jitter_xy, jitter_xy)), float(rng.uniform(-jitter_xy, jitter_xy)))
        cup_yaw = float(rng.uniform(-float(object_cfg["yaw_jitter_rad"]), float(object_cfg["yaw_jitter_rad"])))

        first = int(self.contact_orders[order_index][0])
        return EpisodeSpec(
            episode_id=f"{self.ds['dataset_version']}_{self.split}_{index:06d}",
            dataset_version=str(self.ds["dataset_version"]), split=self.split,
            random_seed=random_seed, region_order=self.region_order,
            target_region=-2, mixed=False, regime=f"case_{case_index + 1}", finger_mode="ambient",
            link_temps_c=temps, fingertip_temps_c=fingertip,
            contact_quality="stable", contact_hold_s=float(self.duration_schedules[schedule_index][first]),
            sensor_rc_s=rc, sensor_noise_c=noise,
            sensor_bias_c=bias, sensor_quantization_c=quant,
            probe_pos=probes[first][0], probe_tool_dir=probes[first][1],
            probe_finger_dir=probes[first][2],
            probe_pre_dist=float(self.task["regions"][self.region_order[first]]["pre_dist"]),
            cup_xy_jitter_m=cup_xy, cup_yaw_jitter_rad=cup_yaw,
            material_parameters=tuple(sorted((str(k), float(v)) for k, v in self._material.items())),
            instruction=self.instruction,
            contact_order=tuple(int(v) for v in self.contact_orders[order_index]),
            contact_hold_s_by_region=self.duration_schedules[schedule_index],
            probe_positions=tuple(p[0] for p in probes),
            probe_tool_dirs=tuple(p[1] for p in probes),
            probe_finger_dirs=tuple(p[2] for p in probes),
        )

    def __iter__(self):
        for i in range(self.count):
            yield self.sample(i)


__all__ = ["SPLITS", "ThermalContactSampler"]

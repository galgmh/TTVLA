"""受控单区域触碰的控制器（不判断温度）。

状态机::

    RESET -> APPROACH -> SEEK_CONTACT -> HOLD -> RETRACT -> DONE
                                     \\-> FAILED（未建立接触）

* RESET      同步控制器到实测位姿，指位保持张开；随后直接进入 APPROACH；
* APPROACH   分四段到位：先升到杯口以上（姿态不变）→ 原地转到探测姿态 →
             水平移到预接触位正上方 → 下降到预接触位；到达后直接开始插入；
* SEEK_CONTACT 先沿接近方向直线插入到接触位，再以 ``contact.close_step_m``
             每策略步闭合手指，直到任一指尖法向力超过
             ``CONTACT_THRESHOLD_N``；失败档只闭合 ``fail_retreat_m``；
* HOLD       把法向力误差积分为指位修正，稳定在 ``spec.target_force_n``；
* RETRACT    张开手指、退回预接触位，停留 ``timing.post_contact_s``。

控制器只读力与位姿，不读温度——温度只被记录，不参与决策。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from sim.control.actions import DeltaPoseController, ProbePose, point_world, tcp_pose

# 判定“碰到”的法向力阈值（N）：明显高于传感器噪声，又远低于目标力。
CONTACT_THRESHOLD_N = 0.2
# 插入阶段沿接近方向的线速度（m/s）。
SEEK_INSERT_SPEED_M_PER_S = 0.08
# 接近路径先升到杯口以上这么高，再水平移动、再下降——避免手臂扫过杯口。
HOVER_CLEARANCE_M = 0.06
# 硬上限：50 s 一集足够覆盖全部阶段。
MAX_POLICY_STEPS = 1000


@dataclass
class ProbeOutcome:
    """一集的结局摘要（写进 HDF5 的 episode 元数据）。"""

    termination_reason: str
    contact_detected: bool
    contact_start_s: float | None
    contact_end_s: float | None
    policy_steps: int
    peak_normal_force_n: float


def _numpy(tensor) -> np.ndarray:
    return np.asarray(tensor.detach().cpu(), dtype=np.float64)


class ProbeController:
    """一集一次的单区域触碰。"""

    def __init__(self, bundle, spec):
        self.bundle = bundle
        self.spec = spec
        task = bundle.cfg
        self.dt = float(task["controller"]["decimation"]) * float(task["controller"]["sim_dt"])
        self.decimation = int(task["controller"]["decimation"])
        self.sample_every = int(task["contact"]["sample_every_sim_steps"])
        self.timing = task["timing"]
        self.contact_cfg = task["contact"]

        self.pose: ProbePose = point_world(
            *bundle.cup_pose(),
            pos_local=spec.probe_pos,
            tool_dir=spec.probe_tool_dir,
            finger_dir=spec.probe_finger_dir,
            pre_dist=spec.probe_pre_dist,
        )
        self.controller = DeltaPoseController(bundle)

        # 四段接近路径：先升到杯口以上（姿态不变），原地转到探测姿态，
        # 再水平移到预接触位正上方，最后下降到预接触位。姿态转换必须在
        # 远离杯子的高点上做，否则手指扫过杯口。
        cup_pos, _ = bundle.cup_pose()
        cup_top = float(cup_pos[2]) + float(bundle.cup_spec.aabb_max[2])
        start_pos, start_quat = tcp_pose(bundle)
        hover_z = max(float(start_pos[2]), cup_top + HOVER_CLEARANCE_M)
        self.approach_waypoints = (
            (np.array([start_pos[0], start_pos[1], hover_z], dtype=np.float64), start_quat),
            (np.array([start_pos[0], start_pos[1], hover_z], dtype=np.float64), self.pose.quat),
            (np.array([self.pose.pre_pos[0], self.pose.pre_pos[1], hover_z], dtype=np.float64), self.pose.quat),
            (np.asarray(self.pose.pre_pos, dtype=np.float64).copy(), self.pose.quat),
        )
        # 第一段只走位置（姿态保持复位姿态），第二段只转姿态，后两段位姿都已
        # 对齐，只需看位置。
        self._approach_check_rot = (False, True, False, False)
        self._approach_stage = 0

        # 手指法向投影方向（link 系）与接触力传感器名。
        self.inner_dirs = [np.asarray(lay.inner_dir, dtype=np.float64) for lay in bundle.layouts]
        self.force_names = [f"tactile_link_force_{name}" for name in bundle.finger_link_names]
        self.bool_names = [f"tactile_link_bool_{name}" for name in bundle.finger_link_names]

        # 最近一次真正下发的指令（TCP 位姿 + 夹爪开口），由 _command 填写，
        # 记录器逐帧取用。
        self.last_command: tuple[np.ndarray, float] | None = None

        self.phase = "RESET"
        self.step_index = 0  # 已执行的策略步数
        self.sim_step = 0  # 已执行的仿真步数
        self.phase_steps = 0  # 当前阶段已执行的策略步数
        self.contact_start_s: float | None = None
        self.contact_end_s: float | None = None
        self.peak_force = 0.0
        self.failure_reason = ""

        self.width_start = float(2.0 * task["robot"]["open_finger_pos"])
        self.width_floor = float(2.0 * task["robot"]["closed_finger_pos"])
        # 失败档的闭合预算：只允许闭合 fail_retreat_m（每根手指）。
        if spec.contact_quality == "failed":
            self.close_stop_width = self.width_start - 2.0 * float(spec.fail_retreat_m)
        else:
            self.close_stop_width = self.width_floor

        self._insert_from: np.ndarray | None = None
        self._insert_steps = 0
        self._insert_done = False

    # ---------------------------------------------------------------- 读数
    def finger_forces(self) -> tuple[np.ndarray, np.ndarray]:
        """返回 (法向力 (2,), 是否接触 (2,))，由 link 系接触力投影得到。"""
        from sim.sensors.tactile import read_all

        data = read_all(self.bundle.link_contacts, temporal_reduction="last")
        normal = np.zeros(2, dtype=np.float64)
        touching = np.zeros(2, dtype=bool)
        for i, (fname, bname) in enumerate(zip(self.force_names, self.bool_names)):
            if fname in data:
                f_link = _numpy(data[fname]).reshape(3)
                # 杯子夹在两指之间：手指受到的压力方向与“指向另一根手指”相反。
                normal[i] = max(0.0, -float(f_link @ self.inner_dirs[i]))
            if bname in data:
                touching[i] = bool(_numpy(data[bname]).reshape(-1)[0] > 0.5)
        return normal, touching

    # ---------------------------------------------------------------- 主循环
    def run(self, on_sample: Callable[[int], None] | None = None) -> ProbeOutcome:
        """执行一整集；每 ``sample_every`` 个仿真步回调一次 ``on_sample``。"""
        self._begin()
        while self.phase not in ("DONE", "FAILED"):
            self._command()
            for _ in range(self.decimation):
                self.bundle.step()
                if on_sample is not None and self.sim_step % self.sample_every == 0:
                    on_sample(self.sim_step)
                self.sim_step += 1
            self.step_index += 1
            self.phase_steps += 1
            normal, touching = self.finger_forces()
            self._update(normal, touching)
            if self.step_index >= MAX_POLICY_STEPS and self.phase not in ("DONE", "FAILED"):
                self._fail("max_steps")
        return ProbeOutcome(
            termination_reason=self.termination_reason,
            contact_detected=self.contact_start_s is not None,
            contact_start_s=self.contact_start_s,
            contact_end_s=self.contact_end_s,
            policy_steps=self.step_index,
            peak_normal_force_n=self.peak_force,
        )

    @property
    def termination_reason(self) -> str:
        if self.phase == "DONE":
            return f"contact_{self.spec.contact_quality}" if self.contact_start_s is not None else "no_contact"
        return self.failure_reason or "running"

    # ---------------------------------------------------------------- 内部
    def _begin(self) -> None:
        self.controller.reset()
        self.controller.set_width_target(self.width_start)
        self.phase = "APPROACH"
        self.phase_steps = 0
        self.step_index = 0
        self.sim_step = 0
        self._approach_stage = 0
        self._insert_done = False

    def _set_phase(self, phase: str) -> None:
        self.phase = phase
        self.phase_steps = 0

    def _fail(self, reason: str) -> None:
        """记录失败原因并滑行到收尾（张开、退回）后结束。"""
        if not self.failure_reason:
            self.failure_reason = reason
        if self.contact_start_s is not None and self.contact_end_s is None:
            self.contact_end_s = self.step_index * self.dt
        if self.phase == "RETRACT":
            self._set_phase("FAILED")
        else:
            self._set_phase("RETRACT")

    def _command(self) -> None:
        ctrl = self.controller
        if self.phase == "APPROACH":
            wp_pos, wp_quat = self.approach_waypoints[self._approach_stage]
            ctrl.set_pose_target(wp_pos, wp_quat)
            ctrl.set_width_target(self.width_start)
        elif self.phase == "SEEK_CONTACT":
            ctrl.set_pose_target(self._insert_target(), self.pose.quat)
            if self._insert_done:
                target = ctrl.width_target - 2.0 * float(self.contact_cfg["close_step_m"])
                ctrl.set_width_target(max(target, self.close_stop_width))
        elif self.phase == "HOLD":
            ctrl.set_pose_target(self.pose.pos, self.pose.quat)
        elif self.phase == "RETRACT":
            ctrl.set_pose_target(self.pose.pre_pos, self.pose.quat)
            ctrl.set_width_target(self.width_start)
        # DONE / FAILED 保持最后指令。
        self.last_command = ctrl.apply()

    def _insert_target(self) -> np.ndarray:
        """插入阶段的 TCP 目标：从预接触位线性推进到接触位。"""
        if self._insert_done:
            return self.pose.pos
        frac = min(1.0, float(self.phase_steps + 1) / float(max(1, self._insert_steps)))
        start = self.pose.pre_pos if self._insert_from is None else self._insert_from
        return np.asarray(start, dtype=np.float64) + frac * (self.pose.pos - start)

    def _update(self, normal: np.ndarray, touching: np.ndarray) -> None:
        self.peak_force = max(self.peak_force, float(normal.max()))
        any_contact = bool(touching.any()) or bool((normal > CONTACT_THRESHOLD_N).any())

        if self.phase == "APPROACH":
            wp_pos, wp_quat = self.approach_waypoints[self._approach_stage]
            check_rot = self._approach_check_rot[self._approach_stage]
            if self.controller.at_pose(
                wp_pos,
                wp_quat if check_rot else None,
                pos_tol=0.006,
                rot_tol=0.08,
            ) and self.controller.settled():
                if self._approach_stage + 1 < len(self.approach_waypoints):
                    self._approach_stage += 1
                    self.phase_steps = 0
                    return
                # 已到预接触位：直接开始插入，不再额外停留。
                self._insert_from = np.asarray(self.controller.cmd_pos, dtype=np.float64).copy()
                distance = float(np.linalg.norm(self.pose.pos - self._insert_from))
                self._insert_steps = max(
                    1, int(np.ceil(distance / (SEEK_INSERT_SPEED_M_PER_S * self.dt)))
                )
                self._insert_done = False
                self._set_phase("SEEK_CONTACT")
            elif self.phase_steps * self.dt >= float(self.contact_cfg["seek_timeout_s"]):
                self._fail("approach_timeout")
            return

        if self.phase == "SEEK_CONTACT":
            if not self._insert_done:
                if self.phase_steps >= self._insert_steps:
                    self._insert_done = True
                    self.phase_steps = 0
                return
            if any_contact and float(normal.max()) >= CONTACT_THRESHOLD_N:
                self.contact_start_s = self.step_index * self.dt
                self._set_phase("HOLD")
                return
            if self.bundle.gripper_width() <= self.close_stop_width + 1e-6:
                self._fail("no_contact")
                return
            if self.phase_steps * self.dt >= float(self.contact_cfg["seek_timeout_s"]):
                self._fail("seek_timeout")
            return

        if self.phase == "HOLD":
            err = float(normal.max()) - float(self.spec.target_force_n)
            gain = float(self.contact_cfg["gain_m_per_n"])
            self.controller.set_width_target(self.controller.width_target + 2.0 * gain * err)
            if self.phase_steps * self.dt >= float(self.spec.contact_hold_s):
                self.contact_end_s = self.step_index * self.dt
                self._set_phase("RETRACT")
            return

        if self.phase == "RETRACT":
            if self.phase_steps * self.dt >= float(self.timing["post_contact_s"]):
                self._set_phase("DONE" if not self.failure_reason else "FAILED")
            return


__all__ = [
    "CONTACT_THRESHOLD_N",
    "MAX_POLICY_STEPS",
    "ProbeController",
    "ProbeOutcome",
    "SEEK_INSERT_SPEED_M_PER_S",
]

"""纯抓取基线控制器：复用 ProbeController 的安全接近，改为夹紧、抬升和保持。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sim.control.probe_controller import CONTACT_THRESHOLD_N, ProbeController


@dataclass(frozen=True)
class GraspPolicyConfig:
    policy_id: str
    instruction: str
    grasp_region: str
    lift_height_m: float
    lift_success_height_m: float
    grasp_timeout_s: float
    lift_timeout_s: float
    lift_hold_s: float
    bilateral_force_n: float


def grasp_policy_from_task(task: dict[str, Any]) -> GraspPolicyConfig:
    """Build one scripted expert from a goal-only task definition."""
    required = str(task["success"]["required_grasp_region"])
    region_by_goal = {"cup_wall": "middle", "handle": "handle", "cup_rim": "upper"}
    if required not in region_by_goal:
        raise ValueError(f"任务 {task['task_id']} 不是单握点直接抓取任务：{required}")
    min_lift = float(task["success"]["min_lift_height_m"])
    return GraspPolicyConfig(
        policy_id=str(task["task_id"]),
        instruction=str(task["instruction"]),
        grasp_region=region_by_goal[required],
        lift_height_m=max(0.12, min_lift + 0.03),
        lift_success_height_m=min_lift,
        grasp_timeout_s=6.0,
        lift_timeout_s=5.0,
        lift_hold_s=0.50,
        bilateral_force_n=0.30,
    )


class GraspLiftController(ProbeController):
    """不读温度的闭环抓取专家。"""

    def __init__(self, bundle, spec, policy: GraspPolicyConfig):
        super().__init__(bundle, spec)
        self.policy = policy
        self.initial_cup_z = float(bundle.cup_pose()[0][2])
        self.lift_target = np.asarray(self.pose.pos, dtype=np.float64).copy()
        self.lift_target[2] += float(policy.lift_height_m)

    @property
    def termination_reason(self) -> str:
        if self.phase == "DONE":
            return "grasp_lift_complete"
        return self.failure_reason or "running"

    def _command(self) -> None:
        if self.phase == "GRASP":
            self.controller.set_pose_target(self.pose.pos, self.pose.quat)
            self.controller.set_width_target(self.width_floor)
            self.last_command = self.controller.apply()
            return
        if self.phase in ("LIFT", "HOLD_LIFT"):
            self.controller.set_pose_target(self.lift_target, self.pose.quat)
            self.last_command = self.controller.apply()
            return
        super()._command()

    def _update(self, normal: np.ndarray, touching: np.ndarray) -> None:
        self.peak_force = max(self.peak_force, float(normal.max(initial=0.0)))
        if self.phase in (
            "APPROACH",
            "SEEK_CONTACT",
            "RETRACT_OPEN",
            "RETRACT_MOVE",
            "RETRACT_SETTLE",
        ):
            super()._update(normal, touching)
            # 父类在建立接触后进入 HOLD；纯抓取改为继续双指夹紧。
            if self.phase == "HOLD":
                self._set_phase("GRASP")
            return

        if self.phase == "GRASP":
            bilateral = bool(np.all(touching)) and bool(np.all(normal >= float(self.policy.bilateral_force_n)))
            if bilateral:
                self._set_phase("LIFT")
                return
            if self.bundle.gripper_width() <= self.width_floor + 1e-6:
                self._fail("grasp_no_bilateral_contact")
                return
            if self.phase_steps * self.dt >= float(self.policy.grasp_timeout_s):
                self._fail("grasp_timeout")
            return

        if self.phase == "LIFT":
            if self.controller.at_pose(self.lift_target, self.pose.quat, pos_tol=0.008, rot_tol=0.10):
                self._set_phase("HOLD_LIFT")
                return
            if self.phase_steps * self.dt >= float(self.policy.lift_timeout_s):
                self._fail("lift_timeout")
            return

        if self.phase == "HOLD_LIFT":
            cup_z = float(self.bundle.cup_pose()[0][2])
            if cup_z - self.initial_cup_z < float(self.policy.lift_success_height_m):
                if self.phase_steps * self.dt >= float(self.policy.lift_hold_s):
                    self._fail("cup_not_lifted")
                return
            if self.phase_steps * self.dt >= float(self.policy.lift_hold_s):
                self.contact_end_s = self.step_index * self.dt
                self._set_phase("DONE")


__all__ = ["GraspLiftController", "GraspPolicyConfig", "grasp_policy_from_task"]


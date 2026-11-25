"""纯抓取基线控制器：复用 ProbeController 的安全接近，改为夹紧、抬升和保持。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sim.control.controller import tcp_pose
from sim.control.probe_controller import ProbeController


@dataclass(frozen=True)
class GraspPolicyConfig:
    policy_id: str
    instruction: str
    grasp_region: str
    lift_height_m: float
    lift_success_height_m: float
    grasp_timeout_s: float
    lift_hold_s: float
    touch_hold_s: float = 0.0
    grasp_hold_s: float = 0.0


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
        lift_hold_s=0.50,
    )


class GraspLiftController(ProbeController):
    """不读温度的闭环抓取专家。"""

    def __init__(self, bundle, spec, policy: GraspPolicyConfig):
        super().__init__(bundle, spec)
        self.policy = policy
        self.initial_cup_z = float(bundle.cup_pose()[0][2])
        self.grasp_pos = np.asarray(self.pose.pos, dtype=np.float64).copy()
        self.grasp_quat = np.asarray(self.pose.quat, dtype=np.float64).copy()
        self.lift_target: np.ndarray | None = None
        self.lift_quat: np.ndarray | None = None
        self.lift_trajectory = None
        self.lift_cursor = 0
        self.grasp_hold_steps = max(1, int(round(float(policy.grasp_hold_s) / self.dt)))
        self.contact_hold_count = 0

    def _begin(self) -> None:
        super()._begin()
        # The cup can move during contact and closure.  Plan LIFT only after
        # bilateral grasp confirmation, from the then-measured TCP pose.
        self.lift_trajectory = None
        self.lift_cursor = 0

    def _begin_at_contact(self) -> None:
        """Continue from an existing closed contact without re-approaching."""
        self.controller.reset()
        self.controller.set_width_target(self.width_floor)
        self.grasp_pos, self.grasp_quat = tcp_pose(self.bundle)
        self.lift_trajectory = None
        self.lift_cursor = 0
        self.phase = "GRASP"
        self.phase_steps = 0
        self.step_index = 0
        self.sim_step = 0
        self.contact_hold_count = 0

    def _plan_lift_from_current(self) -> None:
        """Plan a vertical lift from the measured post-grasp TCP pose."""
        start_pos, start_quat = tcp_pose(self.bundle)
        self.lift_target = np.asarray(start_pos, dtype=np.float64).copy()
        self.lift_target[2] += float(self.policy.lift_height_m)
        self.lift_quat = np.asarray(start_quat, dtype=np.float64).copy()
        arm_q = self.bundle.joint_state()[self.bundle.arm_dofs]
        self.lift_trajectory = self.controller.plan_joint_trajectory(
            (
                (start_pos, start_quat),
                (self.lift_target, self.lift_quat),
            ),
            initial_arm_q=arm_q,
        )
        self.lift_cursor = 0

    @property
    def termination_reason(self) -> str:
        if self.phase == "DONE":
            return "grasp_lift_complete"
        return self.failure_reason or "running"

    def _command(self) -> None:
        if self.phase == "GRASP":
            self.last_command = self.controller.grasp(
                self.grasp_pos, self.grasp_quat, closed_width=self.width_floor
            )
            return
        if self.phase == "LIFT":
            if self.lift_trajectory is None:
                raise RuntimeError("lift trajectory was not planned")
            self.last_command = self.controller.lift_trajectory(
                self.lift_trajectory,
                self.lift_cursor,
                closed_width=self.width_floor,
            )
            return
        if self.phase == "HOLD_LIFT":
            if self.lift_target is None or self.lift_quat is None:
                raise RuntimeError("lift target was not planned")
            self.last_command = self.controller.hold(
                self.lift_target, self.lift_quat, closed_width=self.width_floor
            )
            return
        super()._command()

    def _update(self, normal: np.ndarray, touching: np.ndarray) -> None:
        self.peak_force = max(self.peak_force, float(normal.max(initial=0.0)))
        if self.phase in (
            "APPROACH",
            "SEEK_CONTACT",
            "RETRACT_OPEN",
            "RELEASE_SETTLE",
            "RETRACT_MOVE",
            "RETRACT_SETTLE",
        ):
            super()._update(normal, touching)
            if self.phase == "HOLD":
                self.contact_hold_count = 0
            return

        if self.phase == "HOLD":
            if self.phase_steps * self.dt >= float(self.policy.touch_hold_s):
                self.grasp_pos, self.grasp_quat = tcp_pose(self.bundle)
                self.contact_hold_count = 0
                self._set_phase("GRASP")
            return

        if self.phase == "GRASP":
            bilateral = bool(np.all(touching))
            self.contact_hold_count = self.contact_hold_count + 1 if bilateral else 0
            if self.contact_hold_count >= self.grasp_hold_steps:
                self._plan_lift_from_current()
                self._set_phase("LIFT")
                return
            if self.bundle.gripper_width() <= self.width_floor + 1e-6:
                self._fail("grasp_no_bilateral_contact")
                return
            if self.phase_steps * self.dt >= float(self.policy.grasp_timeout_s):
                self._fail("grasp_timeout")
            return

        if self.phase == "LIFT":
            if self.lift_trajectory is None:
                self._fail("lift_not_planned")
                return
            self.lift_cursor += 1
            if self.lift_cursor >= len(self.lift_trajectory):
                if self.lift_target is None or self.lift_quat is None:
                    self._fail("lift_target_missing")
                    return
                if not self.controller.at_pose(
                    self.lift_target,
                    self.lift_quat,
                    pos_tol=0.03,
                    rot_tol=0.35,
                ):
                    self._fail("lift_tracking_error")
                    return
                self._set_phase("HOLD_LIFT")
                return
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


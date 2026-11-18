"""纯抓取基线控制器：复用 ProbeController 的安全接近，改为夹紧、抬升和保持。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

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


def load_grasp_policy_config(path: str | Path) -> GraspPolicyConfig:
    path = Path(path)
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {field.name for field in GraspPolicyConfig.__dataclass_fields__.values()}
    if set(raw) != required:
        raise ValueError(f"{path} 字段必须恰为 {sorted(required)}，实为 {sorted(raw)}")
    cfg = GraspPolicyConfig(**raw)
    if not cfg.policy_id or not cfg.instruction or not cfg.grasp_region:
        raise ValueError("policy_id / instruction / grasp_region 不能为空")
    for name in (
        "lift_height_m", "lift_success_height_m", "grasp_timeout_s",
        "lift_timeout_s", "lift_hold_s", "bilateral_force_n",
    ):
        if float(getattr(cfg, name)) <= 0.0:
            raise ValueError(f"{name} 必须为正数")
    if cfg.lift_success_height_m > cfg.lift_height_m:
        raise ValueError("lift_success_height_m 不能大于 lift_height_m")
    return cfg


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


__all__ = ["GraspLiftController", "GraspPolicyConfig", "load_grasp_policy_config"]


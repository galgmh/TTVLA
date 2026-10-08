"""7-dim delta-pose action pipeline (LIBERO-compatible semantics).

    a[0:3] * pos_scale_m   -> world-frame TCP position delta (grip center)
    a[3:6] * rot_scale_rad -> world-frame axis-angle orientation delta
    a[6]                   -> gripper synergy: -1 open .. +1 closed

The arm is driven by damped-least-squares IK (warm-started from the current
qpos) targeting the hand link with ``local_point`` = TCP offset; the solved
joint targets go through Genesis' PD position controller. The TCP is the
grip center between the fingertips, so the action/observation semantics match
LIBERO's OSC delta controller even though the implementation is IK-based.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tempvla.rotations import quat_apply_delta, quat_error_rotvec, quat_normalize, quat_rotate
from tempvla.scene import SceneBundle


def tcp_pose(bundle: SceneBundle) -> tuple[np.ndarray, np.ndarray]:
    """Grip-center (TCP) pose in world frame: hand pose + offset along hand z."""
    pos, quat = bundle.hand_pose()
    offset = np.asarray(bundle.cfg.robot.tcp_offset, dtype=np.float64)
    return pos + quat_rotate(quat, offset), quat


def normalized_action_to(
    current_pos: np.ndarray,
    current_quat: np.ndarray,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    grip: float,
    *,
    pos_scale_m: float,
    rot_scale_rad: float,
) -> np.ndarray:
    """Scripted-policy helper: normalized 7-dim action that reduces the pose error."""
    a_pos = np.clip((np.asarray(target_pos) - np.asarray(current_pos)) / pos_scale_m, -1.0, 1.0)
    a_rot = np.clip(quat_error_rotvec(target_quat, current_quat) / rot_scale_rad, -1.0, 1.0)
    return np.concatenate([a_pos, a_rot, [float(np.clip(grip, -1.0, 1.0))]])


@dataclass
class ActionInfo:
    ik_pos_err_m: float
    ik_rot_err_rad: float
    finger_target_m: float
    clipped: bool


class DeltaPoseController:
    """Applies normalized 7-dim actions to the robot; owns the last TCP command.

    Actions are LIBERO-style *deltas*: each step commands the current measured
    TCP pose plus one action delta (re-anchored every control step). The last
    command stays exposed as ``target_pos`` / ``target_quat`` for diagnostics.
    """

    def __init__(self, bundle: SceneBundle):
        self.bundle = bundle
        self.cfg = bundle.cfg.controller
        self.robot = bundle.robot
        self.hand_link = bundle.robot.get_link(bundle.hand_link_name)
        self.tcp_offset = np.asarray(bundle.cfg.robot.tcp_offset, dtype=np.float64)
        self.target_pos = np.zeros(3)
        self.target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.finger_target = float(bundle.cfg.robot.open_finger_pos)
        self.reset()

    # ------------------------------------------------------------------ control

    def reset(self) -> None:
        """Sync setpoints to the actual pose (call after scene.reset())."""
        self.target_pos, self.target_quat = tcp_pose(self.bundle)
        self.finger_target = float(self.bundle.cfg.robot.open_finger_pos)
        self._command_fingers(self.finger_target)

    def step(self, action: np.ndarray) -> ActionInfo:
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        clipped = bool(np.any(np.abs(a[:7]) > 1.0))
        a = np.clip(a[:7], -1.0, 1.0)

        # Anchor on the *measured* TCP pose: integrating an internal setpoint
        # instead lets a lagging (or blocked) arm wind the command ahead of
        # itself — Gate 1's trace showed 70 mm of lead mid-approach, then a
        # 19 mm dive past the goal, a 244 N palm-vs-object impact and a runaway
        # setpoint. Re-anchoring keeps every command within one delta of the
        # arm, so the loop converges monotonically and cannot wind up.
        pos_now, quat_now = tcp_pose(self.bundle)
        self.target_pos = pos_now + a[:3] * self.cfg.pos_scale_m
        self.target_quat = quat_apply_delta(quat_now, a[3:6] * self.cfg.rot_scale_rad)

        q_full, err = self.robot.inverse_kinematics(
            link=self.hand_link,
            pos=self.target_pos,
            quat=self.target_quat,
            local_point=self.tcp_offset,
            init_qpos=self.robot.get_qpos(),
            respect_joint_limit=True,
            max_samples=self.cfg.ik_max_samples,
            max_solver_iters=self.cfg.ik_max_iters,
            damping=self.cfg.ik_damping,
            seed=0,
            return_error=True,
        )
        q_full = _np(q_full).reshape(-1)
        err = _np(err).reshape(-1)[:6]

        q_cmd = q_full[self.bundle.arm_dofs]
        self.robot.control_dofs_position(q_cmd, self.bundle.arm_dofs)

        grip = 0.5 * (a[6] + 1.0)
        r = self.bundle.cfg.robot
        commanded = (1.0 - grip) * r.open_finger_pos + grip * r.closed_finger_pos
        self.finger_target = self._slew(self.finger_target, commanded)
        self._command_fingers(self.finger_target)

        return ActionInfo(
            ik_pos_err_m=float(np.linalg.norm(err[:3])),
            ik_rot_err_rad=float(np.linalg.norm(err[3:])),
            finger_target_m=self.finger_target,
            clipped=clipped,
        )

    def _slew(self, current: float, commanded: float) -> float:
        """Rate-limit the finger setpoint to `finger_speed_m_per_s` (0 = off).

        Measured (docs §9-R22): this slows the physical approach ~10x but does
        not smooth the recorded tactile onset — at the substep the sensor
        registers contact the finger has moved only ~0.01 mm, yet depth steps
        0 -> ~4.5 mm, because the probes float 0.5 mm proud of the pad and the
        SDF query only runs once the solver reports contact.
        """
        speed = self.cfg.finger_speed_m_per_s
        if speed <= 0.0:
            return commanded
        max_delta = speed / self.cfg.control_hz
        return current + float(np.clip(commanded - current, -max_delta, max_delta))

    def _command_fingers(self, target: float) -> None:
        self.robot.control_dofs_position(
            np.full(len(self.bundle.finger_dofs), target, dtype=np.float64), self.bundle.finger_dofs
        )


def _np(value) -> np.ndarray:
    return np.asarray(value.detach().cpu() if hasattr(value, "detach") else value, dtype=np.float64)


__all__ = [
    "ActionInfo",
    "DeltaPoseController",
    "normalized_action_to",
    "tcp_pose",
    "quat_normalize",
]

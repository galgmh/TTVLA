"""Scripted pick expert: normalized 7-dim actions, policy-identical interface.

Phases:

    APPROACH -> DESCEND -> CLOSE -> LIFT -> HOLD -> DONE
                  ^ tactile contact + squeeze before lifting

The expert consumes only the policy-rate observation dict (never the engine),
so it exercises exactly the interface a learned policy would see. Per-episode
jitter (grasp xy offset, approach yaw, closure amount) is sampled on reset to
broaden the demonstration distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from tempvla.actions import normalized_action_to
from tempvla.config import ExpertConfig  # dataclass lives in config.py (YAML-tunable)
from tempvla.rotations import quat_error_rotvec, quat_mul, rotvec_to_quat, yaw_of_quat, yaw_to_quat
from tempvla.scene import SceneBundle

# 180 deg about world x: hand z-axis points down, fingers close horizontally.
TOP_DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])


class Phase(str, Enum):
    APPROACH = "approach"
    DESCEND = "descend"
    CLOSE = "close"
    LIFT = "lift"
    HOLD = "hold"
    DONE = "done"


@dataclass
class ExpertState:
    phase: str = Phase.APPROACH.value
    steps_in_phase: int = 0
    contact_seen: bool = False
    squeeze_count: int = 0
    grasp_confirmed: bool = False
    jitter: dict[str, Any] = field(default_factory=dict)


class PickExpert:
    """Deterministic pick state machine with a small randomized jitter."""

    def __init__(
        self,
        cfg: ExpertConfig | None = None,
        *,
        pos_scale_m: float = 0.05,
        rot_scale_rad: float = 0.5,
    ):
        self.cfg = cfg or ExpertConfig()
        self.pos_scale_m = float(pos_scale_m)
        self.rot_scale_rad = float(rot_scale_rad)
        self.state = ExpertState()
        self.grasp_pos = np.zeros(3)
        self.grasp_quat = TOP_DOWN_QUAT.copy()
        self.target_pos = np.zeros(3)

    # ------------------------------------------------------------------- setup

    def reset(self, bundle: SceneBundle, rng: np.random.Generator | None = None) -> None:
        cfg = self.cfg
        rng = rng or np.random.default_rng()
        xy_jitter = rng.uniform(-cfg.grasp_xy_jitter_m, cfg.grasp_xy_jitter_m, size=2)
        yaw_jitter = float(rng.uniform(-cfg.approach_yaw_jitter_rad, cfg.approach_yaw_jitter_rad))

        obj_pos, obj_quat = bundle.object_pose()
        self.grasp_pos = np.array(
            [obj_pos[0] + xy_jitter[0], obj_pos[1] + xy_jitter[1], obj_pos[2] + cfg.grasp_z_offset_m]
        )
        approach_yaw = yaw_jitter
        if str(bundle.object_spec.shape).lower() == "box":
            # A box is only graspable face-on: the fingers must close along a face
            # normal, or they land on an edge/corner. The jitter picks which face
            # (nearest 90 deg multiple), then the object's own body yaw is added
            # so the fingers end up exactly parallel to that face. Quantizing
            # (object_yaw + jitter) instead would align to the *world* axes and
            # leave a residual equal to the object yaw (up to +-17 deg here),
            # which wedges the pads on an edge: uneven marker field, weak press.
            object_yaw = yaw_of_quat(obj_quat)
            approach_yaw = round(yaw_jitter / (np.pi / 2.0)) * (np.pi / 2.0) + object_yaw
        self.grasp_quat = quat_mul(yaw_to_quat(approach_yaw), TOP_DOWN_QUAT)
        self.target_pos = self.grasp_pos + np.array([0.0, 0.0, cfg.approach_clearance_m])
        self.state = ExpertState(jitter={"xy": xy_jitter.tolist(), "yaw": approach_yaw})

    # -------------------------------------------------------------------- step

    def act(self, obs: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        cfg = self.cfg
        st = self.state
        st.steps_in_phase += 1

        cur_pos = np.asarray(obs["ee_pos"], dtype=np.float64)
        cur_quat = rotvec_to_quat(np.asarray(obs["ee_ori"], dtype=np.float64))
        pos_err = float(np.linalg.norm(self.target_pos - cur_pos))
        rot_err = float(np.linalg.norm(quat_error_rotvec(self.grasp_quat, cur_quat)))

        depth = np.asarray(obs["tactile"].get("depth", []), dtype=np.float64)
        in_contact = bool(np.any(depth > cfg.contact_depth_m)) or bool(np.any(obs["contact"]["link_contact"]))

        grip = -1.0
        if st.phase == Phase.APPROACH.value:
            self.target_pos = self.grasp_pos + np.array([0.0, 0.0, cfg.approach_clearance_m])
            if pos_err < cfg.pos_tol_m and rot_err < cfg.rot_tol_rad:
                self._advance(Phase.DESCEND)

        elif st.phase == Phase.DESCEND.value:
            self.target_pos = self.grasp_pos
            if pos_err < cfg.descend_tol_m:
                self._advance(Phase.CLOSE)

        elif st.phase == Phase.CLOSE.value:
            self.target_pos = self.grasp_pos
            grip = cfg.close_extra
            if in_contact and not st.contact_seen:
                st.contact_seen = True
                st.squeeze_count = 0
            if st.contact_seen:
                st.squeeze_count += 1
                if st.squeeze_count >= cfg.squeeze_steps:
                    st.grasp_confirmed = True
                    self._advance(Phase.LIFT)
            elif st.steps_in_phase >= cfg.close_timeout_steps:
                self._advance(Phase.LIFT)  # retry-free: episode likely fails

        elif st.phase in (Phase.LIFT.value, Phase.HOLD.value):
            self.target_pos = self.grasp_pos + np.array([0.0, 0.0, cfg.lift_height_m])
            grip = cfg.close_extra
            if st.phase == Phase.LIFT.value and pos_err < cfg.lift_tol_m:
                self._advance(Phase.HOLD)

        else:  # DONE: hold the last setpoint
            grip = cfg.close_extra

        action = normalized_action_to(
            cur_pos,
            cur_quat,
            self.target_pos,
            self.grasp_quat,
            grip,
            pos_scale_m=self.pos_scale_m,
            rot_scale_rad=self.rot_scale_rad,
        )
        info = {
            "phase": st.phase,
            "pos_err_m": pos_err,
            "contact": in_contact,
            "grasp_confirmed": st.grasp_confirmed,
        }
        return action, info

    # ----------------------------------------------------------------- helpers

    def _advance(self, phase: Phase) -> None:
        self.state.phase = phase.value
        self.state.steps_in_phase = 0


__all__ = ["ExpertConfig", "Phase", "PickExpert", "TOP_DOWN_QUAT"]

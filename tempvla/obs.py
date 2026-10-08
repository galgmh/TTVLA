"""Policy-rate observation assembly + sim-rate burst payload.

Channels (nested keys map to the HDF5 groups under obs/):

    ee_pos (3,) / ee_ori (3,) / gripper_states (1,) / joint_states (9,)
    link_pose (F,7)     world pos + wxyz quat per finger, in layouts order
    object_pose (N,7)   world pos + wxyz quat per object (format 1.2)
    tactile/{modality}  (sum_f prod(grid_m), F)
    tactile/height_map  (F, H, W) metres — dense resampling of the marker field,
                        written only when ``tactile.height_map.enabled`` (1.3)
    contact/link_contact (L,) bool, contact/link_force (L,3)
    temperature/{fingertip_grid, object_grid, objects_grid, link_temps, object}
                        object_grid/object = the *targeted* object, unchanged;
                        objects_grid (N, *object_grid) covers every object
    agentview_rgb (128,128,3) uint8, eye_in_hand_rgb (128,128,3) uint8

Vector-valued tactile modalities are rotated into the world frame (config
``tactile.world_frame``); ``link_pose`` carries the finger quaternions that
rotation used, so the link-frame field is exactly recoverable. Row counts are
per modality — ``tactile.probe_grid`` may be a per-modality mapping, so a dense
gel marker grid does not widen force/depth (docs §5.2). Temperature channels are
read from the ground-truth branch by default (config
``temperature.record_ground_truth``). The two RGB channels exist only when
``cameras.enabled`` (v2 increment, Gate 3 §8) and are flat under ``obs/`` to
match the LIBERO layout.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tempvla.actions import tcp_pose
from tempvla.rotations import quat_to_matrix, quat_to_rotvec
from tempvla.scene import SceneBundle
from tempvla.sensors.tactile import FEATURES_PER_PROBE, read_all as read_tactile_all
from tempvla.sensors.temperature import read_link_temperatures, read_temperature
from tempvla.tools.height_map import MARKER_MODALITY, pad_normal_planes, synthesize

VECTOR_MODALITIES = frozenset({"force", "force_torque", "elastomer", "proximity"})


def _np(value: Any) -> np.ndarray:
    return np.asarray(value.detach().cpu() if hasattr(value, "detach") else value, dtype=np.float64)


class ObsBuilder:
    """Reads one full observation (policy rate) or a fast burst payload (sim rate)."""

    def __init__(self, bundle: SceneBundle):
        self.bundle = bundle
        self.gt = bundle.cfg.temperature.record_ground_truth

    def _env0(self, arr: np.ndarray) -> np.ndarray:
        return arr[0] if self.bundle.n_envs == 1 and arr.ndim >= 1 and arr.shape[0] == 1 else arr

    # ------------------------------------------------------------------ proprio

    def proprio(self) -> dict[str, np.ndarray]:
        pos, quat = tcp_pose(self.bundle)
        joint = self.bundle.joint_state()
        r = self.bundle.cfg.robot
        finger_mean = float(np.mean(joint[len(self.bundle.arm_dofs) :]))
        span = max(r.open_finger_pos - r.closed_finger_pos, 1e-6)
        grip = 1.0 - 2.0 * (finger_mean - r.closed_finger_pos) / span
        return {
            "ee_pos": pos.astype(np.float32),
            "ee_ori": quat_to_rotvec(quat).astype(np.float32),
            "gripper_states": np.array([grip], dtype=np.float32),
            "joint_states": joint.astype(np.float32),
        }

    # ------------------------------------------------------------------ tactile

    def tactile(
        self, *, mode: str | None = None, modalities: tuple[str, ...] | None = None
    ) -> dict[str, np.ndarray]:
        """One array per modality: ``(n_fingers * prod(grid_m), F)``.

        Row blocks follow ``bundle.tactile_layouts[m]`` (finger order), so a
        channel's row count is its *own* grid's — force/depth stay 12/finger
        while a dense gel field is 256/finger (docs §5.2). ``modalities``
        narrows which sensors are read at all, not just which are returned:
        the 200 Hz burst path must not pay for a modality it discards.
        """
        tcfg = self.bundle.cfg.tactile
        wanted = [m for m in tcfg.modalities if modalities is None or m in modalities]
        handles = [h for h in self.bundle.tactile if h.modality in set(wanted)]
        raw = read_tactile_all(handles, temporal_reduction=mode or tcfg.temporal_reduction)
        out: dict[str, np.ndarray] = {}
        for modality in wanted:
            parts = []
            for lay in self.bundle.tactile_layouts[modality]:
                arr = self._env0(_np(raw[f"tactile_{modality}_{lay.link_name}"]))
                # Strict reshape: a wrong element count must raise here, not
                # silently regroup (the old `-1` did exactly that whenever the
                # count happened to be 2P).
                arr = arr.reshape(lay.num_probes, FEATURES_PER_PROBE[modality])
                if modality in VECTOR_MODALITIES:
                    if tcfg.world_frame:
                        quat = _np(self.bundle.robot.get_link(lay.link_name).get_quat()).reshape(-1)[:4]
                        arr = arr @ quat_to_matrix(quat).T
                elif arr.shape[-1] == 1:
                    arr = arr.reshape(-1)
                parts.append(arr)
            out[modality] = np.concatenate(parts, axis=0).astype(np.float32)
        return out

    def link_pose(self) -> np.ndarray:
        """``(n_fingers, 7)`` f32: world position + wxyz quaternion per finger.

        Recorded as ``obs/link_pose`` so a ``world_frame: true`` file is exactly
        invertible — ``disp_link = disp_world @ R(q)`` recovers the pad-normal
        marker field that is otherwise lost on rotation (docs §9-R8). Rows are
        in ``bundle.layouts`` order, i.e. the order the tactile channels are
        concatenated in, and the quaternions are the ones ``tactile()`` rotates
        with (one simulator state, no stepping in between).
        """
        rows = []
        for lay in self.bundle.layouts:
            link = self.bundle.robot.get_link(lay.link_name)
            pos = _np(link.get_pos()).reshape(-1)[:3]
            quat = _np(link.get_quat()).reshape(-1)[:4]
            rows.append(np.concatenate([pos, quat]))
        return np.stack(rows).astype(np.float32)

    def object_pose(self) -> np.ndarray:
        """``(n_objects, 7)`` f32: world position + wxyz quaternion per object.

        Same encoding and same simulator state as ``link_pose``, so a consumer can
        place every body of a multi-object scene (format 1.2). Row order is
        ``bundle.objects`` = ``cfg.object_list()``.
        """
        return self.bundle.object_poses()

    def height_map(self, tactile: dict[str, np.ndarray], link_pose: np.ndarray) -> np.ndarray | None:
        """``(n_fingers, H, W)`` float32 metres — dense map from the marker field.

        A resampling of the elastomer channel onto a UniVTAC-``press_depth``-like
        image (format 1.3, docs §5.2 / §9-R25); None unless
        ``tactile.height_map.enabled``. The rotation back to the link frame uses
        the very quaternions recorded in ``obs/link_pose`` — the ones a reader
        inverts with — so writer and reader agree up to the interpolation alone.
        Deliberately *not* called from ``fast_payload()``: the 200 Hz burst must
        not resample an image per substep.
        """
        cfg = self.bundle.cfg.tactile.height_map
        tcfg = self.bundle.cfg.tactile
        if not cfg.enabled or MARKER_MODALITY not in tactile:
            return None
        layouts = self.bundle.tactile_layouts[MARKER_MODALITY]
        grid = tcfg.grid_for(MARKER_MODALITY)
        quats = np.asarray(link_pose, dtype=np.float64)[:, 3:7] if tcfg.world_frame else None
        disp = tactile[MARKER_MODALITY].reshape(len(layouts), -1, 3)
        planes = pad_normal_planes(disp, quats, [lay.probe_local_normal for lay in layouts], grid)
        return synthesize(planes, cfg.size, cfg.interpolate)

    def contact(self) -> dict[str, np.ndarray]:
        raw = read_tactile_all(self.bundle.link_contacts, temporal_reduction="last")
        bools, forces = [], []
        for link_name in self.bundle.finger_link_names:
            bools.append(self._env0(_np(raw[f"tactile_link_bool_{link_name}"])).reshape(-1))
            forces.append(self._env0(_np(raw[f"tactile_link_force_{link_name}"])).reshape(3))
        return {
            "link_contact": np.stack(bools).astype(bool),
            "link_force": np.stack(forces).astype(np.float32),
        }

    # -------------------------------------------------------------- temperature

    def temperature(self, *, ground_truth: bool | None = None) -> dict[str, np.ndarray]:
        gt = self.gt if ground_truth is None else ground_truth
        raw = read_temperature(self.bundle.temperatures, ground_truth=gt)
        finger_grids = [
            self._env0(_np(raw[f"temperature_finger_{lay.link_name}"])) for lay in self.bundle.layouts
        ]
        object_grids = [
            self._env0(_np(raw[f"temperature_object_{i}"])) for i in range(len(self.bundle.objects))
        ]
        objects_grid = np.stack(object_grids)
        # `object_grid` / `object` keep their single-object meaning: the object
        # this episode is acting on (format 1.2 adds `objects_grid` for the rest).
        target_grid = object_grids[self.bundle.target_index]
        out = {
            "fingertip_grid": np.stack(finger_grids).astype(np.float32),
            "object_grid": target_grid.astype(np.float32),
            "objects_grid": objects_grid.astype(np.float32),
            "object": np.array([float(target_grid.mean())], dtype=np.float32),
        }
        link_temps = read_link_temperatures(self.bundle.temperatures[-1])
        if link_temps is not None:
            out["link_temps"] = self._env0(_np(link_temps)).astype(np.float32)
        return out

    # ------------------------------------------------------------------- cameras

    def cameras(self) -> dict[str, np.ndarray]:
        """Rendered RGB frames, ``{<name>_rgb: (H,W,3) uint8}``.

        Empty unless ``cameras.enabled``, so the no-camera path is untouched.
        ``SceneBundle.render_cameras()`` returns ``(n_envs,H,W,3)``; with one
        env ``_env0`` drops the batch axis to the per-step ``(H,W,3)`` the
        writer stacks into ``(T,H,W,3)``.

        ``astype`` copies: the renderer's read-back buffer is reused between
        calls, and ``run_episode`` keeps every step's frame until the episode
        ends. Deliberately *not* called from ``fast_payload()`` — the burst
        recorder runs at 200 Hz and would spend its whole budget drawing.
        """
        if not self.bundle.cameras:
            return {}
        return {
            f"{name}_rgb": self._env0(np.asarray(frame)).astype(np.uint8)
            for name, frame in self.bundle.render_cameras().items()
        }

    # ------------------------------------------------------------------- bundles

    def build(self) -> dict[str, Any]:
        obs: dict[str, Any] = {}
        obs.update(self.proprio())
        obs["link_pose"] = self.link_pose()
        obs["object_pose"] = self.object_pose()
        obs["tactile"] = self.tactile()
        height_map = self.height_map(obs["tactile"], obs["link_pose"])
        if height_map is not None:
            obs["tactile"]["height_map"] = height_map
        obs["contact"] = self.contact()
        obs["temperature"] = self.temperature()
        obs.update(self.cameras())
        return obs

    def fast_payload(self) -> dict[str, np.ndarray]:
        """Small sim-rate payload for burst windows (L4).

        ``recording.burst_modalities`` bounds cost: the payload is captured every
        substep, so a dense marker field is filtered out *before* the sensor read
        rather than after (docs §5.5).
        """
        keep = self.bundle.cfg.recording.burst_modalities
        tac = self.tactile(mode="last", modalities=None if keep is None else tuple(keep))
        temp = self.temperature()
        con = self.contact()
        pos, _ = tcp_pose(self.bundle)
        payload = {
            "tcp_pos": pos.astype(np.float32),
            "finger_joint_pos": self.bundle.joint_state()[len(self.bundle.arm_dofs) :].astype(np.float32),
            "contact_bool": con["link_contact"],
            "contact_force": con["link_force"],
            "finger_temp": temp["fingertip_grid"]
            .reshape(len(self.bundle.layouts), -1)
            .mean(axis=1)
            .astype(np.float32),
            "object_temp": temp["object"],
        }
        for modality, arr in tac.items():
            payload[f"fingertip_{modality}"] = arr.astype(np.float32)
        return payload

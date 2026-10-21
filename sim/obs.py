"""Policy-rate observation assembly + sim-rate burst payload.

Channels (nested keys map to the HDF5 groups under obs/):

    ee_pos (3,) / ee_ori (3,) / gripper_states (1,) / joint_states (9,)
    link_pose (F,7)     world pos + wxyz quat per finger, in layouts order
    object_pose (N,7)   world pos + wxyz quat per object (format 1.2)
    tactile/force       (F,6) AT-VLA 6D resultant per finger, link frame:
                        [Fn(3), Ft(3)] (normal + tangential, compression > 0)
    tactile/elastomer   (F*P,2) 2D marker displacement in the pad tangent
                        plane, finger-major (format 1.6)
    contact/link_contact (F,1) bool, contact/link_force (F,3) — the link sensors
                        report (n_envs,1), so the bool channel keeps a trailing
                        singleton axis rather than the (F,) this line used to
                        claim; the force channel is reshaped to 3 and does not
    temperature/{fingertip_grid, object_grid, objects_grid, link_temps, object}
                        object_grid/object = the *targeted* object, unchanged;
                        objects_grid (N, *object_grid) covers every object
    agentview_rgb (H,W,3) uint8, eye_in_hand_rgb (H,W,3) uint8

The derived tactile channels are expressed in the finger/link frame (format
1.6): ``link_pose`` carries the world pose so a consumer can rotate them if it
needs to. Row counts are per modality — the coarse force grid and the dense
16x16 marker grid are distinct derivations over the same pad (docs §5.2).
Temperature channels are
read from the measured branch by default (``RECORD_GROUND_TRUTH = False`` in
:mod:`sim.sensors.temperature`, format 1.5, docs §9-R36): ``fingertip_grid`` is
one lumped measured cell per finger with the official RC/noise/quantization in
it. The two RGB channels exist only when ``cameras.enabled`` (v2 increment,
Gate 3 §8) and are flat under ``obs/`` to match the LIBERO layout.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sim.expert.actions import tcp_pose
from sim.rotations import quat_to_matrix, quat_to_rotvec
from sim.scene import SceneBundle
from sim.sensors.tactile import (
    FEATURES_PER_PROBE,
    TEMPORAL_REDUCTION,
    WORLD_FRAME,
    force_6d,
    grid_for,
    marker_2d,
    read_all as read_tactile_all,
)
from sim.sensors.temperature import read_link_temperatures, read_temperature

# ------------------------------------------------------------------ constants
#
# The camera rendering backend is a portability choice, not a task choice, so
# the YAML does not expose it. ``raster`` is the validated default (Gate 3/4):
# offscreen EGL on any headless Linux box, llvmpipe under WSL2. ``batch`` =
# Madrona, one draw call for all cameras — faster where a Vulkan loader + NVIDIA
# ICD exist, but it aborts ``scene.build()`` on WSL2 (docs §9-R3). ``raytrace``
# needs OptiX. A script that needs another backend overrides this constant
# before ``build_scene()``.
RENDERER = "raster"

VECTOR_MODALITIES = frozenset({"force", "force_torque", "elastomer", "proximity"})


def _np(value: Any) -> np.ndarray:
    return np.asarray(value.detach().cpu() if hasattr(value, "detach") else value, dtype=np.float64)


class ObsBuilder:
    """Reads one full observation (policy rate) or a fast burst payload (sim rate)."""

    def __init__(self, bundle: SceneBundle):
        self.bundle = bundle
        from sim.sensors.temperature import RECORD_GROUND_TRUTH

        self.gt = RECORD_GROUND_TRUTH

    def _env0(self, arr: np.ndarray) -> np.ndarray:
        return arr[0] if self.bundle.n_envs == 1 and arr.ndim >= 1 and arr.shape[0] == 1 else arr

    # ------------------------------------------------------------------ proprio

    def proprio(self) -> dict[str, np.ndarray]:
        pos, quat = tcp_pose(self.bundle)
        joint = self.bundle.joint_state()
        r = self.bundle.cfg["robot"]
        finger_mean = float(np.mean(joint[len(self.bundle.arm_dofs) :]))
        span = max(r["open_finger_pos"] - r["closed_finger_pos"], 1e-6)
        grip = 1.0 - 2.0 * (finger_mean - r["closed_finger_pos"]) / span
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
        """Derived tactile arrays, one entry per modality (format 1.6).

        ``force``     -> ``(n_fingers, 6)``: per-finger sum of the 12 probe
                         forces, decomposed as ``[Fn(3), Ft(3)]`` along the pad
                         normal (AT-VLA 6D force).
        ``elastomer`` -> ``(n_fingers*P, 2)``: each marker's displacement
                         projected onto the pad tangent plane, finger-major.

        Row blocks follow ``bundle.tactile_layouts[m]`` (finger order), so a
        channel's row count is its *own* grid's — the force grid is 12/finger,
        the marker grid 256/finger (docs §5.2). ``modalities`` narrows which
        sensors are read at all, not just which are returned: the 200 Hz burst
        path must not pay for a modality it discards.
        """
        modalities_cfg = tuple(self.bundle.cfg["tactile"]["modalities"])
        wanted = [m for m in modalities_cfg if modalities is None or m in modalities]
        handles = [h for h in self.bundle.tactile if h.modality in set(wanted)]
        raw = read_tactile_all(handles, temporal_reduction=mode or TEMPORAL_REDUCTION)
        out: dict[str, np.ndarray] = {}
        for modality in wanted:
            parts = []
            for lay in self.bundle.tactile_layouts[modality]:
                arr = self._env0(_np(raw[f"tactile_{modality}_{lay.link_name}"]))
                # Strict reshape: a wrong element count must raise here, not
                # silently regroup (the old `-1` did exactly that whenever the
                # count happened to be 2P).
                arr = arr.reshape(lay.num_probes, FEATURES_PER_PROBE[modality])
                if modality == "force":
                    # One 6D resultant per finger, not one row per probe.
                    parts.append(force_6d(arr, lay.inner_dir)[None, :])
                    continue
                if modality == "elastomer":
                    # Project onto the pad tangent plane; the normal component
                    # is intentionally dropped (format 1.6 stores 2D markers).
                    parts.append(
                        marker_2d(arr, lay.probe_local_pos, lay.inner_dir, grid_for(modality))
                    )
                    continue
                if modality in VECTOR_MODALITIES:
                    if WORLD_FRAME:
                        quat = _np(self.bundle.robot.get_link(lay.link_name).get_quat()).reshape(-1)[:4]
                        arr = arr @ quat_to_matrix(quat).T
                elif arr.shape[-1] == 1:
                    arr = arr.reshape(-1)
                parts.append(arr)
            out[modality] = np.concatenate(parts, axis=0).astype(np.float32)
        return out

    def link_pose(self) -> np.ndarray:
        """``(n_fingers, 7)`` f32: world position + wxyz quaternion per finger.

        Format 1.6 records tactile signals in the finger/link frame; this pose
        is the reference a consumer uses to rotate them to world:
        ``p_world = R(q) @ p_link + pos``. Rows follow ``bundle.layouts`` order,
        i.e. the finger order every tactile channel is concatenated in, and are
        read from the same simulator state as the tactile arrays.
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
        ``bundle.objects`` = ``cfg["objects"]``.
        """
        return self.bundle.object_poses()

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
        """Temperature channels; ``ground_truth`` defaults to ``RECORD_GROUND_TRUTH``.

        Format 1.5 (docs §9-R36): the default is the *measured* channel.
        ``fingertip_grid`` is the lumped (T, n_fingers, 1, 1, 1) measured cell
        per finger — the official RC / noise / quantization live in it — while
        the object grids carry no non-idealities, so their ``read()`` equals
        the raw field.
        """
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
        """Rendered camera frames.

        ``{<name>_rgb: (H,W,3) uint8}``.  Empty unless ``cameras.enabled``, so
        the no-camera path is untouched.  ``SceneBundle.render_cameras()``
        returns ``(n_envs,H,W,3)``; with one env ``_env0`` drops the batch axis
        to the per-step ``(H,W,3)`` the writer stacks into ``(T,H,W,3)``.

        ``astype`` copies: the renderer's read-back buffer is reused between
        calls, and ``run_episode`` keeps every step's frame until the episode
        ends. Deliberately *not* called from ``fast_payload()`` — the burst
        recorder runs at 200 Hz and would spend its whole budget drawing.
        """
        out: dict[str, np.ndarray] = {}
        if self.bundle.cameras:
            out.update(
                {
                    f"{name}_rgb": self._env0(np.asarray(frame)).astype(np.uint8)
                    for name, frame in self.bundle.render_cameras().items()
                }
            )
        return out

    # ------------------------------------------------------------------- bundles

    def build(self) -> dict[str, Any]:
        obs: dict[str, Any] = {}
        obs.update(self.proprio())
        obs["link_pose"] = self.link_pose()
        obs["object_pose"] = self.object_pose()
        obs["tactile"] = self.tactile()
        obs["contact"] = self.contact()
        obs["temperature"] = self.temperature()
        obs.update(self.cameras())
        return obs

    def fast_payload(self) -> dict[str, np.ndarray]:
        """Small sim-rate payload for burst windows (L4).

        The payload is captured every substep, so every configured tactile
        modality is packed into it (docs §5.5).
        """
        tac = self.tactile(mode="last")
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

"""EpisodeRunner: drives one episode end to end and records all four layers.

L1 policy-rate channels (obs + actions + rewards/dones), L3 contact events,
L4 burst windows around contact onsets (200 Hz, ring-buffered). The episode
lives in memory once and is flushed by the HDF5Writer immediately, so
generation RAM stays flat.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tempvla.actions import DeltaPoseController
from tempvla.config import TaskConfig
from tempvla.expert import PickExpert
from tempvla.obs import ObsBuilder
from tempvla.recording.events import BurstRecorder, ContactMonitor
from tempvla.recording.hdf5_writer import EpisodeData, HDF5Writer
from tempvla.recording.meta import language_instruction
from tempvla.scene import SceneBundle, sample_episode
from tempvla.sensors.finger_layout import FingerLayout, layouts_to_hdf5_multi
from tempvla.tools.height_map import MARKER_MODALITY


@dataclass
class EpisodeResult:
    data: EpisodeData
    success: bool
    steps: int
    stats: dict[str, Any] = field(default_factory=dict)


def summarize_episode(result: EpisodeResult) -> dict[str, Any]:
    """Success + signal-richness metrics (shared by generate.py and autotune.py)."""
    obs = result.data.obs
    heating_c = 0.0
    temp = obs.get("temperature", {})
    if "fingertip_grid" in temp:
        fg = np.asarray(temp["fingertip_grid"])
        per_finger = fg.reshape(fg.shape[0], fg.shape[1], -1).mean(axis=-1)
        heating_c = float((per_finger[-1] - per_finger[0]).max())
    return {
        "success": bool(result.success),
        "steps": int(result.steps),
        "events": int(result.stats.get("events", 0)),
        "windows": int(result.stats.get("windows", 0)),
        "ik_err_max": float(result.stats.get("ik_err_max", 0.0)),
        "heating_c": heating_c,
        "final_phase": result.stats.get("final_phase"),
        "grasp_confirmed": bool(result.stats.get("grasp_confirmed", False)),
    }


def _flatten(prefix: str, value: Any, out: dict[str, np.ndarray]) -> None:
    if isinstance(value, dict):
        for key, sub in value.items():
            _flatten(f"{prefix}/{key}" if prefix else key, sub, out)
    else:
        out[prefix] = np.asarray(value)


def _nest(flat: dict[str, np.ndarray]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, arr in flat.items():
        node = out
        parts = key.split("/")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = arr
    return out


class EpisodeRunner:
    """Owns the controller / expert / recorder for one built scene."""

    def __init__(
        self,
        bundle: SceneBundle,
        *,
        writer: HDF5Writer | None = None,
        expert: PickExpert | None = None,
        verbose: bool = False,
    ):
        self.bundle = bundle
        self.cfg: TaskConfig = bundle.cfg
        self.controller = DeltaPoseController(bundle)
        self.obs_builder = ObsBuilder(bundle)
        self.expert = expert or PickExpert(
            self.cfg.expert,
            pos_scale_m=self.cfg.controller.pos_scale_m,
            rot_scale_rad=self.cfg.controller.rot_scale_rad,
        )
        self.writer = writer
        self.verbose = verbose

    # ------------------------------------------------------------------- episode

    def run_episode(self, *, episode_seed: int, max_steps: int | None = None) -> EpisodeResult:
        cfg = self.cfg
        max_steps = max_steps or cfg.max_episode_steps
        decimation = cfg.controller.decimation
        rng = np.random.default_rng(episode_seed)

        ep = sample_episode(cfg, rng)
        self.bundle.reset_episode(ep)
        if self.verbose:
            t = int(ep["target_index"])
            print(f"[ep {episode_seed}] target={t} ({self.bundle.object_spec.shape}) "
                  f"temp={ep['temps_c'][t]:.1f}C yaw={ep['yaws'][t]:+.2f} "
                  f"xy=({ep['xy_offsets'][t][0]:+.3f},{ep['xy_offsets'][t][1]:+.3f})")
        # One warm step before anything reads state: it publishes the post-reset
        # link poses into the cached buffers and pushes the seeded object
        # temperature into the sensor read caches.
        self.bundle.step()
        self.controller.reset()
        self.expert.reset(self.bundle, rng)

        monitor = ContactMonitor(list(self.bundle.finger_link_names))
        rec = cfg.recording
        burst = (
            BurstRecorder(
                pre_steps=max(1, int(round(rec.burst_pre_s / cfg.controller.sim_dt))),
                post_steps=max(1, int(round(rec.burst_post_s / cfg.controller.sim_dt))),
                max_windows=rec.burst_max_windows,
            )
            if rec.contact_burst
            else None
        )

        obj_z0 = self.bundle.object_center_height()
        obs_log: dict[str, list[np.ndarray]] = {}
        actions: list[np.ndarray] = []
        rewards: list[float] = []
        dones: list[bool] = []
        ik_err_max = 0.0
        stable = 0
        success = False
        frame = 0
        prev_fast_contact = False
        last_info: dict[str, Any] = {}

        for step in range(max_steps):
            obs = self.obs_builder.build()
            action, info = self.expert.act(obs)
            last_info = info

            flat: dict[str, np.ndarray] = {}
            _flatten("", obs, flat)
            for key, arr in flat.items():
                obs_log.setdefault(key, []).append(arr)
            actions.append(np.asarray(action, dtype=np.float32))

            monitor.update(
                step=step,
                frame=frame,
                time_s=step / cfg.controller.control_hz,
                link_bool=obs["contact"]["link_contact"],
                link_force=obs["contact"]["link_force"],
                depth={"f": obs["tactile"]["depth"]} if "depth" in obs["tactile"] else None,
            )

            action_info = self.controller.step(action)
            ik_err_max = max(ik_err_max, action_info.ik_pos_err_m)

            for _ in range(decimation):
                self.bundle.step()
                frame += 1
                if burst is not None:
                    # Trigger from the sim-rate payload so windows are anchored to
                    # the true substep of contact onset, not the policy step.
                    payload = self.obs_builder.fast_payload()
                    burst.push(frame, payload)
                    fast_contact = bool(np.any(payload["contact_bool"]))
                    if fast_contact and not prev_fast_contact:
                        burst.trigger(frame)
                    prev_fast_contact = fast_contact

            obj_pos, _ = self.bundle.object_pose()
            lifted = float(obj_pos[2] - obj_z0) > cfg.lift_success_m
            stable = stable + 1 if lifted else 0
            success = stable >= cfg.stable_steps
            rewards.append(1.0 if success and stable == cfg.stable_steps else 0.0)
            dones.append(False)
            if success:
                break

        if dones:
            dones[-1] = True
        if burst is not None:
            burst.flush()

        obs_final = {key: np.stack(vals, axis=0) for key, vals in obs_log.items()}
        data = EpisodeData(
            obs=_nest(obs_final),
            actions=np.stack(actions, axis=0),
            rewards=np.asarray(rewards, dtype=np.float32),
            dones=np.asarray(dones, dtype=bool),
            events=self._events(monitor, burst),
            attrs=self._attrs(ep, success, len(actions), monitor, burst, ik_err_max, last_info, episode_seed),
        )
        stats = {
            "success": success,
            "steps": len(actions),
            "windows": 0 if burst is None else burst.num_windows,
            "events": len(monitor.events),
            "ik_err_max": ik_err_max,
            "final_phase": last_info.get("phase"),
            "grasp_confirmed": last_info.get("grasp_confirmed", False),
        }
        if self.writer is not None:
            self.writer.add_episode(data, success=success)
        return EpisodeResult(data=data, success=success, steps=len(actions), stats=stats)

    # ------------------------------------------------------------------- helpers

    @staticmethod
    def _events(monitor: ContactMonitor, burst: BurstRecorder | None) -> dict[str, Any]:
        events: dict[str, Any] = {"contact_events": monitor.table()}
        if burst is not None and burst.num_windows:
            windows = {}
            meta = burst.metadata()
            for i, stacked in enumerate(burst.stack()):
                entry = {k: v.astype(np.float32) if v.dtype.kind == "f" else v for k, v in stacked.items()}
                entry["trigger_frame"] = np.array([meta["trigger_frame"][i]], dtype=np.int32)
                entry["start_frame"] = np.array([meta["start_frame"][i]], dtype=np.int32)
                entry["end_frame"] = np.array([meta["end_frame"][i]], dtype=np.int32)
                windows[f"window_{i}"] = entry
            events["burst"] = windows
        return events

    def _attrs(self, ep, success, steps, monitor, burst, ik_err_max, info, episode_seed) -> dict[str, Any]:
        b = self.bundle
        t = int(b.target_index)
        spec = b.object_spec
        specs = b.object_specs
        # Table-center-relative slot each object was placed at (place_xy + jitter).
        # For a single-object scene this equals the legacy `object_xy`.
        slots = np.asarray(ep["xy_offsets"], dtype=np.float32) + np.asarray(
            [o.place_xy for o in b.cfg.object_list()], dtype=np.float32
        )
        poses = b.object_poses()  # (N, 7) pos_world + wxyz quat
        return {
            "success": bool(success),
            "episode_seed": int(episode_seed),
            "num_windows": 0 if burst is None else int(burst.num_windows),
            "num_contact_events": len(monitor.events),
            "ik_err_max_m": float(ik_err_max),
            "expert_final_phase": str(info.get("phase", "")),
            "grasp_confirmed": bool(info.get("grasp_confirmed", False)),
            # `object_*` scalars keep their meaning: the *targeted* object, i.e.
            # the one the expert acted on and the success check watched.
            "object_material": spec.material,
            "object_shape": spec.shape,
            "object_radius_m": spec.radius_m,
            "object_height_m": spec.height_m,
            "object_mass_kg": spec.mass_kg,
            "object_temp_c": float(ep["temps_c"][t]),
            "object_xy": slots[t],
            "object_yaw": float(ep["yaws"][t]),
            "object_final_pos": poses[t, :3],
            "object_link_idx": int(b.obj.links[0].idx),
            "language_instruction": language_instruction(b.cfg, spec.material, spec.shape),
            # format 1.2: the whole scene, so the non-targeted body a video shows
            # moving is accounted for in the file.
            "target_index": t,
            "objects_shape": [s.shape for s in specs],
            "objects_xy": slots,
            "objects_yaw": np.asarray(ep["yaws"], dtype=np.float32),
            "objects_temp_c": np.asarray(ep["temps_c"], dtype=np.float32),
            "objects_final_pos": poses[:, :3],
            "objects_link_idx": np.asarray([int(e.links[0].idx) for e in b.objects], dtype=np.int32),
            "objects_radius_m": np.asarray([s.radius_m for s in specs], dtype=np.float32),
            "objects_height_m": np.asarray([s.height_m for s in specs], dtype=np.float32),
            "objects_mass_kg": np.asarray([s.mass_kg for s in specs], dtype=np.float32),
            "episode_steps": int(steps),
        }

    # --------------------------------------------------------------- static meta

    def file_layouts(self) -> dict[str, Any]:
        """Static geometry written once per file (layouts/ group).

        One layout set per distinct probe grid; the primary modality's grid is
        also written at the root as the legacy alias, and ``modality_grids``
        maps each modality to its group (docs §5.2).
        """
        b = self.bundle
        tcfg = b.cfg.tactile
        modality_grids = {m: tcfg.grid_for(m) for m in tcfg.modalities}
        groups: dict[tuple[int, int], list[FingerLayout]] = {}
        for modality, grid in modality_grids.items():
            groups.setdefault(grid, b.tactile_layouts[modality])
        attrs: dict[str, Any] = {
            # Plain Python strings/lists: h5py attrs accept those directly.
            "finger_link_names": list(b.finger_link_names),
            "hand_link_name": b.hand_link_name,
            # Which frame the vector modalities (force/proximity/elastomer) were
            # rotated into when recorded. Consumers recover the link frame with
            # obs/link_pose; `link_pose_encoding` says how that row is laid out.
            "world_frame": bool(tcfg.world_frame),
            "link_pose_encoding": "pos_world(3) + quat_wxyz(4), float32",
            "modalities": list(tcfg.modalities),
            # Legacy single-value attr = the first object's link; every
            # tracked indenter link (all objects) is in object_link_indices.
            "object_link_idx": int(b.objects[0].links[0].idx),
            "object_link_indices": np.asarray([int(e.links[0].idx) for e in b.objects], dtype=np.int32),
            "tcp_offset": np.asarray(b.cfg.robot.tcp_offset, dtype=np.float32),
            "temperature_grids": np.asarray(
                [*b.cfg.temperature.finger_grid, *b.cfg.temperature.object_grid], dtype=np.int32
            ),
        }
        if tcfg.height_map.enabled:
            attrs["height_map"] = json.dumps(
                {
                    "source": "elastomer",
                    "grid": list(modality_grids.get(MARKER_MODALITY, ())),
                    "size": list(tcfg.height_map.size),
                    "interpolation": tcfg.height_map.interpolate,
                    "unit": "m",
                    # The raster spans the grid's *cells*, half a cell of flat
                    # margin beyond the outermost markers — the same extent model
                    # the renderer uses (gel_render.grid_m_per_px), so a stored
                    # map and a map re-derived from the markers cover the same
                    # patch of pad.
                    "axes": "rows = grid axis 0, cols = grid axis 1; cell-centred, spanning grid rows x cols cells",
                }
            )
        return layouts_to_hdf5_multi(
            groups,
            default_grid=tcfg.grid_for(tcfg.primary_modality),
            modality_grids=modality_grids,
            attrs=attrs,
        )

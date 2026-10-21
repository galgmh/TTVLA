"""EpisodeRunner: drives one episode end to end and records all four layers.

L1 policy-rate channels (obs + actions + rewards/dones), L3 contact events,
L4 burst windows around contact onsets (200 Hz, ring-buffered). The episode
lives in memory once and is flushed by the HDF5Writer immediately, so
generation RAM stays flat.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sim.expert.actions import DeltaPoseController
from sim.config import TaskConfig, language_templates
from sim.expert.expert import Phase, PickExpert, build_expert
from sim.expert.pipeline import PIPELINE, PipelineConfig, target_kind
from sim.obs import ObsBuilder
from sim.recording.events import BurstRecorder, ContactMonitor
from sim.recording.hdf5_writer import EpisodeData, HDF5Writer
from sim.recording.meta import language_instruction
from sim.scene import SceneBundle, sample_episode
from sim.sensors.finger_layout import FingerLayout, layouts_to_hdf5_multi
from sim.sensors.tactile import WORLD_FRAME, channel_format, grid_for, primary_modality
from sim.sensors.temperature import (
    NOISE,
    RECORD_GROUND_TRUTH,
    RESOLUTION,
    SENSOR_TIME_CONSTANT,
)


@dataclass
class EpisodeResult:
    data: EpisodeData
    success: bool
    steps: int
    stats: dict[str, Any] = field(default_factory=dict)


def summarize_episode(result: EpisodeResult) -> dict[str, Any]:
    """Success + signal-richness metrics (used by generate.py)."""
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
        pipeline: PipelineConfig | None = None,
        verbose: bool = False,
    ):
        self.bundle = bundle
        self.cfg: TaskConfig = bundle.cfg
        self.controller = DeltaPoseController(bundle)
        self.obs_builder = ObsBuilder(bundle)
        if expert is None:
            # The task YAML names the goal (`expert.approach`); build_expert maps
            # it to the expert class. An unknown approach raises instead of
            # silently falling back to a plain picker.
            expert = build_expert(
                self.cfg["expert"],
                pipeline=pipeline,
                pos_scale_m=self.cfg["controller"]["pos_scale_m"],
                rot_scale_rad=self.cfg["controller"]["rot_scale_rad"],
            )
        self.expert = expert
        # The pipeline actually in force, for success checks that depend on an
        # expert constant (the open-grip threshold).
        self.pipeline = pipeline or getattr(expert, "pipe", PIPELINE)
        self.writer = writer
        self.verbose = verbose

    # ------------------------------------------------------------------- episode

    def run_episode(self, *, episode_seed: int, max_steps: int | None = None) -> EpisodeResult:
        cfg = self.cfg
        max_steps = max_steps or cfg["max_episode_steps"]
        decimation = cfg["controller"]["decimation"]
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
        # Sensor warm-up (docs §9-R36). The engine zeroes every sensor's measured
        # ring on reset (sensor_manager.py), so a fingertip RC filter would spend
        # its first ~5*tau climbing 0 -> ambient *inside* the recording — an
        # engine init artifact, not sensor realism (measured 2026-10-13: the
        # recorded channel reads ~0.7 C at step 0 and needs ~0.75 s to converge).
        # The true finger field is flat at ambient here, so this settles sensor
        # state, not physics.
        tau = float(SENSOR_TIME_CONSTANT)
        if tau > 0.0 and self.bundle.temperatures:
            for _ in range(int(np.ceil(5.0 * tau / cfg["controller"]["sim_dt"]))):
                self.bundle.step()
        self.controller.reset()
        self.expert.reset(self.bundle, rng)

        monitor = ContactMonitor(list(self.bundle.finger_link_names))
        rec = cfg["recording"]
        burst = (
            BurstRecorder(
                pre_steps=max(1, int(round(rec["burst_pre_s"] / cfg["controller"]["sim_dt"]))),
                post_steps=max(1, int(round(rec["burst_post_s"] / cfg["controller"]["sim_dt"]))),
                max_windows=rec["burst_max_windows"],
            )
            if rec["contact_burst"]
            else None
        )

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
                time_s=step / cfg["controller"]["control_hz"],
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

            ok = self._placed_ok(obs)
            stable = stable + 1 if ok else 0
            success = stable >= cfg["stable_steps"]
            rewards.append(1.0 if success and stable == cfg["stable_steps"] else 0.0)
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

    def _place_candidates(self) -> list[int]:
        """Body indices the instruction accepts as "the requested object in the tray".

        The scene has one body per instruction target; the episode's own
        ``target_index`` is the authority.
        """
        return [int(self.bundle.target_index)]

    def _placed_ok(self, obs: dict[str, Any]) -> bool:
        """``place_in_tray`` success: the targeted body is at rest in the cavity.

        Four conditions, all read from the scene rather than from the expert's
        own belief (docs §9-R39):

        * inside the cavity footprint — the check that does the real work, since
          the tray stands on the table and the table is the same plane the other
          bodies sit on;
        * its bottom on the cavity floor (within ``tray_rest_tol_m``);
        * the pads open — a body held at exactly that height is *also* resting on
          the floor, so the gripper state is what separates "placed" from "still
          held";
        * and the expert has finished (DONE), so the recording carries the whole
          retreat instead of stopping at the instant the body touched down.

        Only the episode's target is tested (:meth:`_place_candidates`). The
        bottom is read from the engine's world AABB, so a body that settled
        tilted or on its side is still accepted.
        """
        b = self.bundle
        cfg = b.cfg
        tray = b.tray
        if tray is None:
            return False
        if str(getattr(self.expert.state, "phase", "")) != Phase.DONE.value:
            return False
        grip = float(np.asarray(obs["gripper_states"], dtype=np.float64).reshape(-1)[0])
        if grip > float(self.pipeline.release_open_tol):
            return False
        for i in self._place_candidates():
            entity = b.objects[i]
            # The engine's own world AABB: its min z is the body's true lowest
            # point whatever the body's orientation. The old upright rule
            # (`pos_z - height/2`) rejected a can that settled leaning on the
            # tray wall: its centre sits higher than half its height, so the
            # predicted bottom fell below the tray floor.
            aabb = np.asarray(entity.get_AABB().detach().cpu(), dtype=np.float64).reshape(-1, 3)
            lo, hi = aabb[0], aabb[1]
            if (
                abs(float(lo[2]) - float(tray.floor_z)) <= float(cfg["tray_rest_tol_m"])
                and tray.contains_xy(float(0.5 * (lo[0] + hi[0])), float(0.5 * (lo[1] + hi[1])))
            ):
                return True
        return False

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
        # Table-center-relative slot each object was placed at (slot + jitter). A
        # task whose slots are per-episode (`slots` in the episode dict, docs
        # §9-R38) must read them from there: re-deriving from `place_xy` would
        # report the canonical layout a permuted episode did not use.
        slot_xy = ep.get("slots")
        if slot_xy is None:
            slot_xy = [o["place_xy"] for o in b.cfg["objects"]]
        slots = np.asarray(ep["xy_offsets"], dtype=np.float32) + np.asarray(slot_xy, dtype=np.float32)
        poses = b.object_poses()  # (N, 7) pos_world + wxyz quat
        attrs = {
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
            "language_instruction": language_instruction(
                b.cfg,
                spec.material,
                spec.shape,
                # Several phrasings rotate by episode seed; one template is
                # `seed % 1 == 0`.
                template_index=episode_seed % len(language_templates(b.cfg)),
            ),
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
        attrs.update(self._temperature_judge_attrs())
        attrs.update(self._place_attrs())
        return attrs

    def _place_attrs(self) -> dict[str, Any]:
        """Where the tray was and what the placements aimed at (docs §9-R39).

        Empty for a task without a tray, so every existing file's attrs are
        unchanged. The tray's numbers are build-time geometry, not a sampled
        pose. Every candidate is judged, so every match is carried: ``places``
        lists all placements in order (``place_bodies``, one JSON record per
        carried body), while ``place_body``/``place_material``/``place_rest_z``
        keep the *last* one for readers of the single-placement layout. Neither
        is necessarily the episode's target -- a judge error shows up as a
        mismatch between the two, alongside ``judge`` -- and an episode where
        every candidate was rejected ends with no placement at all.
        """
        b = self.bundle
        tray = b.tray
        if tray is None:
            return {}
        places = [dict(p) for p in (getattr(self.expert.state, "places", []) or [])]
        place = dict(getattr(self.expert.state, "place", {}) or {}) or (
            places[-1] if places else {}
        )
        out: dict[str, Any] = {
            "success_criterion": str(b.cfg["success_criterion"]),
            "tray_center_xy": np.asarray(tray.center_xy, dtype=np.float32),
            "tray_inner_xy_m": np.asarray([tray.inner_x_m, tray.inner_y_m], dtype=np.float32),
            "tray_floor_z": float(tray.floor_z),
            "tray_rim_z": float(tray.rim_z),
        }
        if places:
            out["place_bodies"] = json.dumps(places)
            out["place_count"] = len(places)
        if place:
            out["place_body"] = int(place["body"])
            out["place_material"] = str(place["material"])
            out["place_rest_z"] = float(place["rest_z"])
        return out

    def _temperature_judge_attrs(self) -> dict[str, Any]:
        """Per-episode temperature-judgement block, empty for other tasks.

        One record per touch: the body, each fingertip's pre-contact baseline, the
        reading and trend at the step the decision fired (measured channel only),
        the threshold, and the verdict. The *label* (`is_target`, `correct`) is
        added here where the scene's target is known; the expert never read it.
        `correct` is None for a touch with no registered contact, which is not
        evidence about the body.
        """
        b = self.bundle
        cfg_e = b.cfg["expert"]
        if not bool(cfg_e["temperature_judge_enabled"]):
            return {}
        log = list(getattr(self.expert.state, "temperature_log", []) or [])
        t = int(b.target_index)
        order = list(getattr(self.expert, "order", None) or [])
        lo_c, hi_c = cfg_e["temperature_judge_threshold_c"]
        out: dict[str, Any] = {
            "temperature_candidates": [int(i) for i in cfg_e["temperature_candidate"]],
            "temperature_candidate_order": "random",
            "temperature_judge_first_body": int(order[0]) if order else None,
            # h5py attrs cannot hold None inside a numeric list (object dtype ->
            # "no native HDF5 equivalent"), so the open ends are written as
            # -inf/+inf. The JSON records below keep them as null.
            "temperature_judge_threshold_c": [
                -math.inf if lo_c is None else float(lo_c),
                math.inf if hi_c is None else float(hi_c),
            ],
            "temperature_judge_target_kind": target_kind(cfg_e["temperature_judge_threshold_c"]),
            # The pipeline the expert actually ran with, not the module default:
            # `EpisodeRunner` accepts a custom one for experiments.
            "temperature_judge_trend_window_s": float(self.pipeline.temperature_trend_window_s),
            "temperature_judge_trend_min_k_per_s": float(
                self.pipeline.temperature_trend_min_k_per_s
            ),
            "temperature_judge_baseline_s": float(self.pipeline.temperature_judge_baseline_s),
            "temperature_judge_read_tail_s": float(self.pipeline.temperature_read_tail_s),
        }
        enriched = []
        for rec in log:
            rec = dict(rec)
            body = rec.get("body")
            rec["is_target"] = bool(body == t)
            rec["correct"] = (
                None
                if rec.get("verdict_is_target") is None
                else bool(bool(rec["verdict_is_target"]) == (body == t))
            )
            enriched.append(rec)
        if enriched:
            out["temperature_judge"] = json.dumps(enriched)
            out["temperature_judge_touches"] = len(enriched)
            out["temperature_judge_correct"] = all(
                r["correct"] for r in enriched if r.get("correct") is not None
            )
        return out

    # --------------------------------------------------------------- static meta

    def file_layouts(self) -> dict[str, Any]:
        """Static geometry written once per file (layouts/ group).

        One layout set per distinct probe grid; the primary modality's grid is
        also written at the root as the legacy alias, and ``modality_grids``
        maps each modality to its group (docs §5.2).
        """
        b = self.bundle
        modalities = tuple(b.cfg["tactile"]["modalities"])
        modality_grids = {m: grid_for(m) for m in modalities}
        groups: dict[tuple[int, int], list[FingerLayout]] = {}
        for modality, grid in modality_grids.items():
            groups.setdefault(grid, b.tactile_layouts[modality])
        formats = {
            m: channel_format(m, len(b.finger_link_names), grid_for(m))
            for m in modalities
        }
        attrs: dict[str, Any] = {
            # Plain Python strings/lists: h5py attrs accept those directly.
            "finger_link_names": list(b.finger_link_names),
            "hand_link_name": b.hand_link_name,
            # Format 1.6 records the derived signals in the finger/link frame.
            # `channel_formats` declares each channel's per-finger shape,
            # feature count and frame; `link_pose` is the world reference.
            "world_frame": bool(WORLD_FRAME),
            "link_pose_encoding": "pos_world(3) + quat_wxyz(4), float32",
            "modalities": list(modalities),
            "channel_formats": json.dumps({m: f for m, f in formats.items() if f is not None}),
            # Legacy single-value attr = the first object's link; every
            # tracked indenter link (all objects) is in object_link_indices.
            "object_link_idx": int(b.objects[0].links[0].idx),
            "object_link_indices": np.asarray([int(e.links[0].idx) for e in b.objects], dtype=np.int32),
            "tcp_offset": np.asarray(b.cfg["robot"]["tcp_offset"], dtype=np.float32),
            # The finger triple is the *actual* grid: read it from the live
            # handles so a stale cfg value cannot drift from what the sensors
            # were built with (format 1.5 lumped (1,1,1), docs §9-R36).
            "temperature_grids": np.asarray(
                [
                    *(b.temperatures[0].grid_size if b.temperatures else b.cfg["temperature"]["finger_grid"]),
                    *b.cfg["temperature"]["object_grid"],
                ],
                dtype=np.int32,
            ),
        }
        # Format 1.5: the official sensor model, self-describing (docs
        # §9-R36). The non-idealities apply to the finger sensors' measured
        # channel only; the object grids record the raw field (their read()
        # equals the ground truth).
        attrs["temperature_sensor"] = json.dumps(
            {
                "grid": list(
                    b.temperatures[0].grid_size if b.temperatures else b.cfg["temperature"]["finger_grid"]
                ),
                "record": "ground_truth" if RECORD_GROUND_TRUTH else "measured",
                "sensor_time_constant": SENSOR_TIME_CONSTANT,
                "noise": NOISE,
                "resolution": RESOLUTION,
                "applies_to": "finger sensors (measured channel); object grids carry none",
            }
        )
        return layouts_to_hdf5_multi(
            groups,
            default_grid=grid_for(primary_modality(modalities)),
            modality_grids=modality_grids,
            attrs=attrs,
        )

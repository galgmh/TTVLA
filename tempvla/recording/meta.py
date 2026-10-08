"""File-level metadata payloads: env_args, problem_info, language instruction.

Everything here must be JSON-serializable (the writer stores these as JSON
strings on the data/ group), and it should be sufficient for a downstream
training script to reconstruct the exact observation/action contract without
reading the generator source.
"""

from __future__ import annotations

from typing import Any

from tempvla.config import TaskConfig


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def language_instruction(cfg: TaskConfig, material: str, shape: str | None = None) -> str:
    """Fill the task template; unknown placeholders are left untouched.

    ``shape`` is required by mixed-object templates (``"lift the {material} {shape}"``)
    and ignored by single-shape ones. Callers that know which object the episode
    targets pass it (the per-demo ``language_instruction`` attr); the file-level
    string uses the first object's shape.
    """
    template = cfg.language_template
    fields = {"material": material}
    if shape is not None:
        fields["shape"] = shape
    try:
        return template.format(**fields)
    except (KeyError, IndexError, ValueError):
        out = template.replace("{material}", material)
        if shape is not None:
            out = out.replace("{shape}", shape)
        return out


def _objects_block(cfg: TaskConfig, bundle: Any | None) -> list[dict[str, Any]]:
    """Per-object descriptions for env_args (one per ``cfg.object_list()`` entry)."""
    specs = list(getattr(bundle, "object_specs", []) or []) if bundle is not None else []
    out: list[dict[str, Any]] = []
    for i, o in enumerate(cfg.object_list()):
        entry: dict[str, Any] = {
            "shape": o.shape,
            "radius_m": list(o.radius_m),
            "height_m": list(o.height_m),
            "mass_kg": list(o.mass_kg),
            "temp_c": list(o.temp_c),
            "place_xy": list(o.place_xy),
            "friction": 1.0,
        }
        if i < len(specs):
            spec = specs[i]
            entry.update(
                {
                    "material": spec.material,
                    "resolved_shape": spec.shape,
                    "resolved_radius_m": float(spec.radius_m),
                    "resolved_height_m": float(spec.height_m),
                    "resolved_mass_kg": float(spec.mass_kg),
                }
            )
        out.append(entry)
    return out


def build_env_args(cfg: TaskConfig, *, bundle: Any | None = None) -> dict[str, Any]:
    """Static description of the environment / observation / action contract."""
    try:
        import genesis as gs

        engine_version = str(getattr(gs, "__version__", "unknown"))
        assets_dir = str(gs.utils.get_assets_dir())
    except Exception:  # noqa: BLE001 - only used for metadata
        engine_version, assets_dir = "unknown", ""

    env: dict[str, Any] = {
        "sim": {
            "engine": "genesis",
            "engine_version": engine_version,
            "assets_dir": assets_dir,
            "dt": cfg.controller.sim_dt,
            "decimation": cfg.controller.decimation,
            "control_hz": cfg.controller.control_hz,
        },
        "robot": {
            "model": "franka_panda",
            "urdf": cfg.robot.urdf,
            "gripper": "parallel_jaw",
            "arm_joint_names": list(cfg.robot.arm_joint_names),
            "finger_joint_names": list(cfg.robot.finger_joint_names),
            "hand_link": cfg.robot.hand_link_name,
            "finger_links": list(cfg.robot.finger_link_names),
            "tcp_offset": list(cfg.robot.tcp_offset),
            "init_qpos": list(cfg.robot.init_qpos),
            "open_finger_pos": cfg.robot.open_finger_pos,
            "closed_finger_pos": cfg.robot.closed_finger_pos,
        },
        "action_space": {
            "type": "delta_tcp_pose_with_gripper",
            "dim": 7,
            "pos_scale_m": cfg.controller.pos_scale_m,
            "rot_scale_rad": cfg.controller.rot_scale_rad,
            "gripper": "synergy -1 = open .. +1 = closed",
            "controller": "ik_position_pd (per-step IK on the hand link local_point=tcp_offset)",
        },
        "observation": {
            "proprio": ["ee_pos", "ee_ori", "gripper_states", "joint_states"],
            "ee_ori_encoding": "axis-angle (rotvec, wxyz-compatible shortest arc)",
            "link_pose": {
                "shape": "(n_fingers, 7) float32",
                "encoding": "world position (3) + wxyz quaternion (4) per finger link",
                "purpose": "exact inverse of tactile.world_frame: disp_link = disp_world @ R(q)",
            },
            "object_pose": {
                "shape": "(n_objects, 7) float32",
                "encoding": "world position (3) + wxyz quaternion (4) per object, same as link_pose",
                "row_order": "env_args.objects",
            },
            "tactile": {
                "layout": "per-modality probe grid per finger link, concatenated finger-major",
                "modalities": list(cfg.tactile.modalities),
                # The single-grid form is kept for readers that predate per-modality
                # grids: it is the *primary* modality's grid, not every channel's.
                "probe_grid": list(cfg.tactile.grid_for(cfg.tactile.primary_modality)),
                "per_modality_probe_grid": {m: list(cfg.tactile.grid_for(m)) for m in cfg.tactile.modalities},
                "rows_per_modality": {
                    m: len(cfg.robot.finger_link_names) * cfg.tactile.grid_for(m)[0] * cfg.tactile.grid_for(m)[1]
                    for m in cfg.tactile.modalities
                },
                "world_frame": cfg.tactile.world_frame,
                "temporal_reduction": cfg.tactile.temporal_reduction,
            },
            "contact": {"link_contact": "bool per finger link", "link_force": "N, 3 per finger link"},
            "temperature": {
                # object_grid/object describe the *targeted* object; objects_grid
                # (format 1.2) stacks every object's grid in env_args.objects order.
                "channels": ["fingertip_grid", "object_grid", "objects_grid", "link_temps", "object"],
                "units": "celsius",
                "ground_truth": cfg.temperature.record_ground_truth,
            },
        },
        "sensors": {
            "tactile": {
                "probe_radius_m": cfg.tactile.probe_radius_m,
                "probe_offset_m": cfg.tactile.probe_offset_m,
                "imperfection": cfg.tactile.imperfection,
                "contact_links": cfg.tactile.contact_links,
            },
            "temperature": {
                "finger_grid": list(cfg.temperature.finger_grid),
                "object_grid": list(cfg.temperature.object_grid),
                "ambient_temp_c": cfg.temperature.ambient_temp_c,
                "convection_coefficient": cfg.temperature.convection_coefficient,
                "sensor_time_constant": cfg.temperature.sensor_time_constant,
                "contact_depth_weight": cfg.temperature.contact_depth_weight,
                "simulate_all_link_temperatures": cfg.temperature.simulate_all_link_temperatures,
            },
        },
        # Legacy single-object block: describes the *first* object (identical to
        # `object` for configs that predate the `objects:` list).
        "object": {
            "shape": cfg.object_list()[0].shape,
            "radius_m": list(cfg.object_list()[0].radius_m),
            "height_m": list(cfg.object_list()[0].height_m),
            "mass_kg": list(cfg.object_list()[0].mass_kg),
            "friction": 1.0,
            "material_sampled_per_file": True,
        },
        # format 1.2: one entry per body, in obs/object_pose row order. Ranges are
        # the config's; `resolved_*` are the values this file's scene froze at
        # build time (same sampling rule as the legacy object block).
        "objects": _objects_block(cfg, bundle),
        # `color` is render-only (it changes the RGB channels' appearance and
        # nothing else), so it is recorded for self-description but omitted when
        # null — runs made before the key existed stay byte-identical.
        "table": {
            "pos": list(cfg.table.pos),
            "size": list(cfg.table.size),
            "height_m": cfg.table.height_m,
            **({"color": list(cfg.table.color)} if cfg.table.color is not None else {}),
        },
        "cameras": {
            "enabled": bool(cfg.cameras.enabled),
            # The validator reads this list to decide what the file must contain,
            # so it has to be exactly what obs.py emits (empty when disabled).
            "channels": ["agentview_rgb", "eye_in_hand_rgb"] if cfg.cameras.enabled else [],
            "renderer": cfg.cameras.renderer,
            "resolution": list(cfg.cameras.resolution),
            "encoding": "uint8 RGB, flat under obs/ (LIBERO layout)",
            "agentview": {
                "pos": list(cfg.cameras.agentview_pos),
                "lookat": list(cfg.cameras.agentview_lookat),
                "fov": cfg.cameras.fov,
            },
            "eye_in_hand": {
                "attached_to": cfg.robot.hand_link_name,
                # Pose in the hand frame (same convention as scene.add_camera).
                "pos": list(cfg.cameras.eye_in_hand_pos),
                "lookat": list(cfg.cameras.eye_in_hand_lookat),
                "up": list(cfg.cameras.eye_in_hand_up),
                "fov": cfg.cameras.eye_in_hand_fov,
            },
            "near": cfg.cameras.near,
            "far": cfg.cameras.far,
        },
        "recording": {
            "burst_hz": 1.0 / cfg.controller.sim_dt,
            "burst_pre_s": cfg.recording.burst_pre_s,
            "burst_post_s": cfg.recording.burst_post_s,
            "burst_max_windows": cfg.recording.burst_max_windows,
        },
        "success": {"lift_success_m": cfg.lift_success_m, "stable_steps": cfg.stable_steps},
        "max_episode_steps": cfg.max_episode_steps,
    }
    if bundle is not None:
        env["robot"]["resolved_joint_names"] = list(bundle.joint_names)
        env["robot"]["resolved_link_names"] = list(bundle.link_names)
    return _jsonable(env)


def build_problem_info(cfg: TaskConfig, material: str) -> dict[str, Any]:
    """Per-file task description (one material per file).

    The scalar ``language_instruction`` / ``object_*`` fields describe the *first*
    object and keep their meaning for single-object files; multi-object files
    carry one entry per body under ``objects``, and every demo additionally
    stores the instruction for the object it actually targeted
    (``language_instruction`` attr).
    """
    objs = cfg.object_list()
    first = objs[0]
    instructions = [language_instruction(cfg, material, o.shape) for o in objs]
    return _jsonable(
        {
            "task_id": cfg.task_id,
            "language_instruction": instructions[0],
            "language_instructions": list(dict.fromkeys(instructions)),
            "language_template": cfg.language_template,
            "object_material": material,
            "material_properties": dict(cfg.material_properties.get(material, {})),
            "all_material_properties": dict(cfg.material_properties),
            "object_temp_c_range": list(first.temp_c),
            "ambient_temp_c": cfg.temperature.ambient_temp_c,
            "object_count": len(objs),
            "objects": [
                {
                    "shape": o.shape,
                    "radius_m": list(o.radius_m),
                    "height_m": list(o.height_m),
                    "mass_kg": list(o.mass_kg),
                    "temp_c": list(o.temp_c),
                    "place_xy": list(o.place_xy),
                    "language_instruction": language_instruction(cfg, material, o.shape),
                }
                for o in objs
            ],
            # Which object an episode targets, and who picks it.
            "target_selection": "per_episode_uniform" if len(objs) > 1 else "single_object",
            "randomization": {
                "per_file": ["object_material", "object_count", "object_radius_m", "object_height_m", "object_mass_kg"],
                "per_episode": [
                    "object_xy_offset_m",
                    "object_yaw_rad",
                    "object_temperature_c",
                    "target_index",
                ],
                "object_xy_jitter_m": first.xy_jitter_m,
                "object_yaw_jitter_rad": first.yaw_jitter_rad,
                "objects_xy_jitter_m": [o.xy_jitter_m for o in objs],
                "objects_yaw_jitter_rad": [o.yaw_jitter_rad for o in objs],
            },
        }
    )


__all__ = ["build_env_args", "build_problem_info", "language_instruction"]

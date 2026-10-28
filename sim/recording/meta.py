"""File-level metadata payloads: env_args, problem_info, language instruction.

Everything here must be JSON-serializable (the writer stores these as JSON
strings on the data/ group), and it should be sufficient for a downstream
training script to reconstruct the exact observation/action contract without
reading the generator source.
"""

from __future__ import annotations

from typing import Any, Sequence

from sim.config import TaskConfig, language_templates, tray_floor_z, tray_rim_z
from sim.expert.pipeline import PIPELINE, PipelineConfig, target_kind
from sim.sensors.tactile import (
    CONTACT_LINKS,
    ELASTOMER_PARAMS,
    PROBE_OFFSET_M,
    PROBE_RADIUS_M,
    TEMPORAL_REDUCTION,
    WORLD_FRAME,
    channel_format,
    grid_for,
    primary_modality,
)
from sim.sensors.temperature import (
    CONTACT_AREA_POLICY,
    CONTACT_DEPTH_WEIGHT,
    CONVECTION_COEFFICIENT,
    NOISE,
    RECORD_GROUND_TRUTH,
    RESOLUTION,
    SENSOR_TIME_CONSTANT,
    SIMULATE_ALL_LINK_TEMPERATURES,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def _mesh_scale_json(value: float | tuple[float, ...]) -> float | list[float]:
    """Scalar stays a float (the historic schema); a per-axis scale is a list."""
    return float(value) if isinstance(value, (int, float)) else [float(v) for v in value]


def language_instruction(
    cfg: TaskConfig,
    material: str,
    shape: str | None = None,
    template_index: int = 0,
) -> str:
    """Fill the task template; unknown placeholders are left untouched.

    ``shape`` is required by mixed-object templates (``"lift the {material} {shape}"``)
    and ignored by single-shape ones. Callers that know which object the episode
    targets pass it (the per-demo ``language_instruction`` attr); the file-level
    string uses the first object's shape.

    ``template_index`` picks one of several phrasings (``language_template`` as a
    YAML list). The runner draws it as ``episode_seed % n``, which spreads the
    phrasings evenly over a file and reproduces the same sentence on a re-run with
    the same seed; the index wraps, so callers need no bounds check.
    """
    templates = language_templates(cfg)
    template = templates[int(template_index) % len(templates)]
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
    """Per-object descriptions for env_args (one per ``cfg["objects"]`` entry)."""
    specs = list(getattr(bundle, "object_specs", []) or []) if bundle is not None else []
    out: list[dict[str, Any]] = []
    for i, o in enumerate(cfg["objects"]):
        shape = str(o["shape"])
        entry: dict[str, Any] = {
            "shape": shape,
            "mass_kg": list(o["mass_kg"]),
            "temp_c": list(o["temp_c"]),
            "place_xy": list(o["place_xy"]),
            "friction": 1.0,
            # Primitive size range: a cube keeps a single side length, the other
            # primitives a radius/height pair (see sim/config.py's shape keys).
            **(
                {"radius_m": list(o["radius_m"]), "height_m": list(o["height_m"])}
                if "radius_m" in o
                else {}
            ),
            **({"side_m": list(o["side_m"])} if "side_m" in o else {}),
            # Render-only surface color, same convention as `table.color` below:
            # recorded for self-description, omitted when unset — every config that
            # does not set one keeps the env_args it had before the key existed.
            **({"color": list(o["color"])} if "color" in o else {}),
            # Mesh bodies: which asset, how to read it (units + axis convention).
            # Omitted for primitives, so their records are unchanged.
            **(
                {
                    "mesh_file": str(o["mesh_file"]),
                    "mesh_scale": _mesh_scale_json(o["mesh_scale"]),
                    "mesh_euler_deg": list(o["mesh_euler_deg"]),
                }
                if shape.lower() == "mesh"
                else {}
            ),
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

    n_fingers = len(cfg["robot"]["finger_link_names"])
    modalities = tuple(cfg["tactile"]["modalities"])
    tactile_formats = {
        m: channel_format(m, n_fingers, grid_for(m)) for m in modalities
    }
    # Late-bound: sim.obs is the single source for the renderer (a script may
    # swap it before building the scene), so read it where it is used.
    from sim.obs import RENDERER

    controller = cfg["controller"]
    robot_cfg = cfg["robot"]
    cameras_cfg = cfg["cameras"]
    recording_cfg = cfg["recording"]
    temp_cfg = cfg["temperature"]
    table_cfg = cfg["table"]
    tray_cfg = cfg["tray"]
    objects_cfg = cfg["objects"]

    env: dict[str, Any] = {
        "sim": {
            "engine": "genesis",
            "engine_version": engine_version,
            "assets_dir": assets_dir,
            "dt": controller["sim_dt"],
            "decimation": controller["decimation"],
            "control_hz": controller["control_hz"],
        },
        "robot": {
            "model": "franka_panda",
            "urdf": robot_cfg["urdf"],
            "gripper": "parallel_jaw",
            "arm_joint_names": list(robot_cfg["arm_joint_names"]),
            "finger_joint_names": list(robot_cfg["finger_joint_names"]),
            "hand_link": robot_cfg["hand_link_name"],
            "finger_links": list(robot_cfg["finger_link_names"]),
            "tcp_offset": list(robot_cfg["tcp_offset"]),
            "init_qpos": list(robot_cfg["init_qpos"]),
            "open_finger_pos": robot_cfg["open_finger_pos"],
            "closed_finger_pos": robot_cfg["closed_finger_pos"],
        },
        "action_space": {
            "type": "delta_tcp_pose_with_gripper",
            "dim": 7,
            "pos_scale_m": controller["pos_scale_m"],
            "rot_scale_rad": controller["rot_scale_rad"],
            "gripper": "synergy -1 = open .. +1 = closed",
            "controller": "ik_position_pd (per-step IK on the hand link local_point=tcp_offset)",
        },
        "observation": {
            "proprio": ["ee_pos", "ee_ori", "gripper_states", "joint_states"],
            "ee_ori_encoding": "axis-angle (rotvec, wxyz-compatible shortest arc)",
            "link_pose": {
                "shape": "(n_fingers, 7) float32",
                "encoding": "world position (3) + wxyz quaternion (4) per finger link",
                "purpose": (
                    "reference for the link-frame tactile signals (format 1.6): "
                    "p_world = R(q) @ p_link + pos"
                ),
            },
            "object_pose": {
                "shape": "(n_objects, 7) float32",
                "encoding": "world position (3) + wxyz quaternion (4) per object, same as link_pose",
                "row_order": "env_args.objects",
            },
            "tactile": {
                "layout": (
                    "force: one 6D resultant per finger (link frame); "
                    "elastomer: per-marker 2D displacement in the pad tangent plane, "
                    "finger-major (format 1.6)"
                ),
                "modalities": list(modalities),
                # The single-grid form is kept for readers that predate per-modality
                # grids: it is the *primary* modality's grid, not every channel's.
                "probe_grid": list(grid_for(primary_modality(modalities))),
                "per_modality_probe_grid": {m: list(grid_for(m)) for m in modalities},
                # Per-channel recorded shape (excluding T). force = (n_fingers, 6),
                # elastomer = (n_fingers*P, 2). Legacy probe channels have no entry.
                "channel_shapes": {
                    m: fmt["shape"] for m, fmt in tactile_formats.items() if fmt is not None
                },
                "channel_formats": {
                    m: {"features": fmt["features"], "frame": fmt["frame"]}
                    for m, fmt in tactile_formats.items()
                    if fmt is not None
                },
                "world_frame": WORLD_FRAME,
                "temporal_reduction": TEMPORAL_REDUCTION,
            },
            "contact": {"link_contact": "bool per finger link", "link_force": "N, 3 per finger link"},
            "temperature": {
                # object_grid/object describe the *targeted* object; objects_grid
                # (format 1.2) stacks every object's grid in env_args.objects order.
                "channels": ["fingertip_grid", "object_grid", "objects_grid", "link_temps", "object"],
                "units": "celsius",
                # Format 1.5 (docs §9-R36): one lumped measured cell per finger —
                # fingertip_grid is (n_fingers, 1, 1, 1) with the official RC /
                # noise / quantization of env_args.sensors.temperature in the
                # data; the object grids record the raw field.
                "fingertip_grid": "lumped measured cell per finger (format 1.5; docs §9-R36)",
                "ground_truth": RECORD_GROUND_TRUTH,
            },
        },
        "sensors": {
            "tactile": {
                "probe_radius_m": PROBE_RADIUS_M,
                "probe_offset_m": PROBE_OFFSET_M,
                "elastomer_params": dict(ELASTOMER_PARAMS),
                "contact_links": CONTACT_LINKS,
            },
            "temperature": {
                # The finger triple is the live grid (the lumped (1,1,1) cell
                # in format 1.5, docs §9-R36); prefer the handle over cfg so
                # the two cannot drift.
                "finger_grid": list(
                    bundle.temperatures[0].grid_size
                    if bundle is not None and getattr(bundle, "temperatures", None)
                    else temp_cfg["finger_grid"]
                ),
                "object_grid": list(temp_cfg["object_grid"]),
                "ambient_temp_c": temp_cfg["ambient_temp_c"],
                "convection_coefficient": CONVECTION_COEFFICIENT,
                # Format 1.5 non-idealities (docs §9-R36): applied to the
                # finger sensors' measured channel only; the object grids
                # carry 0/0/0 and record the raw field.
                "sensor_time_constant": SENSOR_TIME_CONSTANT,
                "noise": NOISE,
                "resolution": RESOLUTION,
                "contact_depth_weight": CONTACT_DEPTH_WEIGHT,
                # Which contact-area口径 fed the heat kernel: "engine" = the
                # engine's per-contact polygon / pi*sum(depth) (docs §9-R33),
                # "pad_face" = the compat rewrite A_i = A_face/sqrt(n_i) (docs
                # §9-R37). Shapes are unchanged either way; the key is what lets
                # a reader tell two otherwise-identical files apart.
                "contact_area_policy": CONTACT_AREA_POLICY,
                "simulate_all_link_temperatures": SIMULATE_ALL_LINK_TEMPERATURES,
            },
        },
        # Legacy single-object block: describes the *first* object (identical to
        # `object` for configs that predate the `objects:` list).
        "object": {
            "shape": objects_cfg[0]["shape"],
            **(
                {
                    "radius_m": list(objects_cfg[0]["radius_m"]),
                    "height_m": list(objects_cfg[0]["height_m"]),
                }
                if "radius_m" in objects_cfg[0]
                else {"side_m": list(objects_cfg[0]["side_m"])}
            ),
            "mass_kg": list(objects_cfg[0]["mass_kg"]),
            "friction": 1.0,
            "material_sampled_per_file": True,
        },
        # format 1.2: one entry per body, in obs/object_pose row order. Ranges are
        # the config's; `resolved_*` are the values this file's scene froze at
        # build time (same sampling rule as the legacy object block).
        "objects": _objects_block(cfg, bundle),
        # Temperature profiles are permuted within each group every episode; the
        # per-demo attrs (objects_temp_c, temperature_profile_sources) hold the
        # actual assignment.
        "temperature_groups": [list(group) for group in cfg["temperature_groups"]],
        "temperature_profile_note": (
            "objects[i].temp_c is a profile, not a fixed body temperature. Profiles "
            "are permuted within each temperature_groups entry per episode; the body "
            "that receives target_index's profile is the target."
        ),
        # `color` is render-only (it changes the RGB channels' appearance and
        # nothing else), so it is recorded for self-description but omitted when
        # null — runs made before the key existed stay byte-identical.
        "table": {
            "pos": list(table_cfg["pos"]),
            "size": list(table_cfg["size"]),
            "height_m": table_cfg["height_m"],
            **({"color": list(table_cfg["color"])} if "color" in table_cfg else {}),
        },
        "cameras": {
            "enabled": bool(cameras_cfg["enabled"]),
            # The validator reads this list to decide what the file must contain,
            # so it has to be exactly what obs.py emits (empty when disabled).
            "channels": (
                ["agentview_rgb", *(["eye_in_hand_rgb"] if cameras_cfg["eye_in_hand"] else [])]
                if cameras_cfg["enabled"]
                else []
            ),
            "renderer": RENDERER,
            "resolution": list(cameras_cfg["resolution"]),
            "encoding": "uint8 RGB, flat under obs/ (LIBERO layout)",
            "agentview": {
                "pos": list(cameras_cfg["agentview_pos"]),
                "lookat": list(cameras_cfg["agentview_lookat"]),
                "fov": cameras_cfg["fov"],
            },
            "eye_in_hand": {
                "attached_to": robot_cfg["hand_link_name"],
                # Pose in the hand frame (same convention as scene.add_camera).
                "pos": list(cameras_cfg["eye_in_hand_pos"]),
                "lookat": list(cameras_cfg["eye_in_hand_lookat"]),
                "up": list(cameras_cfg["eye_in_hand_up"]),
                "fov": cameras_cfg["eye_in_hand_fov"],
            },
            "near": cameras_cfg["near"],
            "far": cameras_cfg["far"],
        },
        "recording": {
            "burst_hz": 1.0 / controller["sim_dt"],
            "burst_pre_s": recording_cfg["burst_pre_s"],
            "burst_post_s": recording_cfg["burst_post_s"],
            "burst_max_windows": recording_cfg["burst_max_windows"],
        },
        "success": {
            "stable_steps": cfg["stable_steps"],
            # Tasks before put_in_tray all succeed by lifting, and their env_args
            # must stay byte-identical, so the criterion and the tray are recorded
            # only when a tray is actually there (docs §9-R39).
            **(
                {
                    "criterion": cfg["success_criterion"],
                    "tray_center_xy": list(tray_cfg["place_xy"]),
                    "tray_inner_xy_m": [tray_cfg["inner_x_m"], tray_cfg["inner_y_m"]],
                    "tray_rest_tol_m": cfg["tray_rest_tol_m"],
                }
                if tray_cfg["enabled"]
                else {}
            ),
        },
        "max_episode_steps": cfg["max_episode_steps"],
    }
    if tray_cfg["enabled"]:
        # Static geometry of the five fixed parts, in the table's frame (docs
        # §9-R39): enough for a consumer to draw the tray without the mesh.
        env["tray"] = {
            "enabled": True,
            "place_xy": list(tray_cfg["place_xy"]),
            "inner_x_m": tray_cfg["inner_x_m"],
            "inner_y_m": tray_cfg["inner_y_m"],
            "rim_m": tray_cfg["rim_m"],
            "wall_m": tray_cfg["wall_m"],
            "floor_m": tray_cfg["floor_m"],
            "color": None if tray_cfg.get("color") is None else list(tray_cfg["color"]),
            "parts": "floor plate + 4 walls (5 fixed boxes; genesis has no compound primitive)",
            "world_center_xy": [
                table_cfg["pos"][0] + tray_cfg["place_xy"][0],
                table_cfg["pos"][1] + tray_cfg["place_xy"][1],
            ],
            "floor_z": tray_floor_z(tray_cfg, table_cfg["height_m"]),
            "rim_z": tray_rim_z(tray_cfg, table_cfg["height_m"]),
        }
    if bundle is not None:
        env["robot"]["resolved_joint_names"] = list(bundle.joint_names)
        env["robot"]["resolved_link_names"] = list(bundle.link_names)
    return _jsonable(env)


def build_problem_info(
    cfg: TaskConfig,
    material: str | Sequence[str],
    *,
    pipeline: PipelineConfig | None = None,
) -> dict[str, Any]:
    """Per-file task description (one material per object).

    The scalar ``language_instruction`` / ``object_*`` fields describe the *first*
    object and keep their meaning for single-object files; multi-object files
    carry one entry per body under ``objects``, and every demo additionally
    stores the instruction for the object it actually targeted
    (``language_instruction`` attr).

    ``material`` may be a sequence, one material per object: the file's scene
    uses exactly those materials, positionally.

    ``pipeline`` is the pipeline the job's expert runs with (the module default
    when omitted), so the recorded knob values describe the run, not a constant.

    A config with several ``language_template`` phrasings adds
    ``language_templates`` (the full list) and ``language_instruction_variants``
    (every distinct rendered sentence). ``language_template`` /
    ``language_instruction`` keep meaning the first phrasing.
    """
    pipe = pipeline or PIPELINE
    objs = cfg["objects"]
    first = objs[0]
    # One name covers every body, exactly as in build_scene (every legacy call);
    # a sequence gives one material per body.
    given = [material] if isinstance(material, str) else list(material)
    if len(given) == 1:
        mats = given * len(objs)
    elif len(given) == len(objs):
        mats = given
    else:
        raise ValueError(f"{len(given)} materials for {len(objs)} objects: {given}")
    groups = [list(group) for group in cfg["temperature_groups"]]
    target = int(cfg["target_index"])
    target_group = next((group for group in groups if target in group), None)
    templates = language_templates(cfg)
    instructions = [
        language_instruction(cfg, mats[i], str(o["shape"])) for i, o in enumerate(objs)
    ]
    # Every sentence that can appear in this file: all phrasings x all bodies.
    # `language_instructions` keeps its historic meaning (one per object, first
    # phrasing, aligned with `objects`); this is an additive superset.
    variants = list(
        dict.fromkeys(
            language_instruction(cfg, mats[i], str(o["shape"]), template_index=t)
            for t in range(len(templates))
            for i, o in enumerate(objs)
        )
    )
    per_episode = ["object_xy_offset_m", "object_yaw_rad", "object_temperature_c"]
    if target_group is not None:
        per_episode += ["temperature_profile_assignment", "target_index"]
    info: dict[str, Any] = {
        "task_id": cfg["task_id"],
        "language_instruction": instructions[0],
        "language_instructions": list(dict.fromkeys(instructions)),
        "language_template": templates[0],
        "language_templates": list(templates),
        "language_instruction_variants": variants,
        "object_material": mats[0],
        "material_properties": dict(cfg["material_properties"].get(mats[0], {})),
        "all_material_properties": dict(cfg["material_properties"]),
        "object_temp_c_range": list(first["temp_c"]),
        "ambient_temp_c": cfg["temperature"]["ambient_temp_c"],
        "object_count": len(objs),
        "objects": [
            {
                "shape": o["shape"],
                # A cube's size is one number (side_m); the other primitives and
                # the mesh bodies keep the radius/height pair.
                **(
                    {"radius_m": list(o["radius_m"]), "height_m": list(o["height_m"])}
                    if "radius_m" in o
                    else {"side_m": list(o["side_m"])}
                ),
                "mass_kg": list(o["mass_kg"]),
                "temp_c": list(o["temp_c"]),
                "place_xy": list(o["place_xy"]),
                "language_instruction": instructions[i],
                **(  # render-only, omitted when unset (see _objects_block)
                    {"color": list(o["color"])} if "color" in o else {}
                ),
                **(  # mesh bodies only; omitted for primitives (see _objects_block)
                    {
                        "mesh_file": str(o["mesh_file"]),
                        "mesh_scale": _mesh_scale_json(o["mesh_scale"]),
                        "mesh_euler_deg": list(o["mesh_euler_deg"]),
                    }
                    if str(o["shape"]).lower() == "mesh"
                    else {}
                ),
            }
            for i, o in enumerate(objs)
        ],
        # Which object an episode targets, and who picks it. `target_index` is a
        # profile anchor when it is inside a temperature group: the body that
        # receives that profile is the episode target.
        "target_selection": (
            "temperature_profile_permutation" if target_group is not None else "pinned"
        ),
        "temperature_groups": groups,
        "temperature_profile_note": (
            "objects[i].temp_c is a temperature profile, not a fixed body temperature. "
            "Within each temperature_groups entry the profiles are permuted per episode; "
            "the body that receives object target_index's profile is the target. The "
            "per-demo objects_temp_c and temperature_profile_sources attrs record the "
            "actual seed for each episode."
        ),
        "randomization": {
            "per_file": ["object_material", "object_count", "object_radius_m", "object_height_m", "object_mass_kg"],
            "per_episode": per_episode,
            "object_xy_jitter_m": first["xy_jitter_m"],
            "object_yaw_jitter_rad": first["yaw_jitter_rad"],
            "objects_xy_jitter_m": [o["xy_jitter_m"] for o in objs],
            "objects_yaw_jitter_rad": [o["yaw_jitter_rad"] for o in objs],
        },
    }
    if len(given) > 1:
        info["objects_material"] = mats
    if cfg["expert"]["temperature_judge_enabled"]:
        cands = [int(i) for i in cfg["expert"]["temperature_candidate"]]
        if len(cands) > 1:
            info["randomization"]["per_episode"].append("temperature_candidate_order")
        thresholds = cfg["expert"]["temperature_judge_threshold_c"]
        interval = [None if v is None else float(v) for v in thresholds]
        kind = target_kind(thresholds)
        info["temperature_judgement"] = {
            "protocol": "touch_all_candidates_event_driven",
            "statistic": (
                "measured fingertip temperature on every control step of the hold: a target as "
                "soon as any touching finger's reading (mean of the last `read_tail_s`) lies "
                "inside the configured [low, high] interval and its trend (least-squares slope "
                "over `trend_window_s` of its post-onset samples) does not carry the reading "
                "out of that interval within one window -- a chilled pad climbing through a "
                "cold interval towards a hot body is a crossing, not arrival (a finger whose "
                "trend window is not full yet cannot match); a rejection as "
                "soon as no touching finger can still reach that interval -- every finger "
                "with a trend so far is either not moving toward it at `trend_min_k_per_s` "
                "(a flat pad counts as not moving) or inside it but about to leave, so one "
                "pad's 'no' cannot veto another pad's 'not yet'. There is no fixed hold "
                "window."
            ),
            "sensor_channel": "obs/temperature/fingertip_grid (measured; RC/noise/quantised)",
            "candidates": cands,
            "candidate_order": "random",
            "target_interval_c": interval,
            "target_kind": kind,
            "trend_window_s": float(pipe.temperature_trend_window_s),
            "trend_min_k_per_s": float(pipe.temperature_trend_min_k_per_s),
            "baseline_s": float(pipe.temperature_judge_baseline_s),
            "read_tail_s": float(pipe.temperature_read_tail_s),
            "note": (
                "The pads close on a candidate and the measured fingertip decides, event by "
                "event, whether it matches the target interval; a body that does not is "
                "released from the touch pose and the next candidate is tried, in a fresh "
                "per-episode permutation. Every candidate is judged: a match is carried to "
                "the tray and the machine then returns for the bodies it has not felt yet, "
                "so every matching candidate is carried. Nothing is force-committed -- a "
                "candidate that never answers the interval is left on the table even when it "
                "is the last one. Nothing here reads the scene's ground-truth object "
                "temperature. Success still requires the targeted body in the tray at DONE, "
                "so a judge error fails the episode. Per-demo attrs: "
                "temperature_candidates, temperature_candidate_order, temperature_judge_* and "
                "`temperature_judge` (JSON, one record per touch: body, per-finger "
                "baseline/measured/delta/slope, reason, hold_s, verdict_is_target, committed). "
                "The target is the body that received the configured target profile in the "
                "per-episode temperature-group permutation, so its slot is not fixed."
            ),
        }
    if cfg["tray"]["enabled"]:
        t = cfg["tray"]
        info["placement"] = {
            "success_criterion": cfg["success_criterion"],
            "tray": {
                "inner_x_m": t["inner_x_m"],
                "inner_y_m": t["inner_y_m"],
                "rim_m": t["rim_m"],
                "wall_m": t["wall_m"],
                "floor_m": t["floor_m"],
                "place_xy": list(t["place_xy"]),
                "floor_z": tray_floor_z(t, cfg["table"]["height_m"]),
                "rim_z": tray_rim_z(t, cfg["table"]["height_m"]),
            },
            "tolerance": {
                "rest_tol_m": cfg["tray_rest_tol_m"],
                "inside_footprint": "the body centre must lie inside the inner rectangle",
                "pads": "must read open (gripper_states[0] <= release_open_tol)",
            },
            "note": (
                "Every candidate of the temperature judge is tried; every candidate that "
                "answers the target interval is carried to a tray slot and released on the "
                "cavity floor, so an episode can place zero, one or several bodies (the first "
                "match takes the cavity centre, later ones step aside). The episode ends once "
                "the last candidate has been judged, and success is read from the scene (the "
                "targeted body at rest inside the cavity with the pads open, held for "
                "stable_steps after the expert finished), never from the expert's own "
                "belief. Per-demo attrs: place_bodies (JSON, one record per placement), "
                "place_count, place_body, place_material, place_rest_z and the tray geometry; "
                "place_body/place_material/place_rest_z describe the last placement."
            ),
        }
    return _jsonable(info)


__all__ = ["build_env_args", "build_problem_info", "language_instruction"]

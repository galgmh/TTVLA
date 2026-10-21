"""Task configuration: a validated YAML mapping.

One YAML file describes one task end to end (robot, table, objects, sensors,
thermal parameters, randomization ranges, success criterion, recording). The
YAML is the single source of truth: :func:`load_task_config` returns a plain
nested ``dict`` and every module reads ``cfg["section"]["key"]`` directly.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

# A task config: nested plain mappings/lists, exactly as the YAML spells them.
TaskConfig = dict[str, Any]

_TOP_REQUIRED = frozenset(
    {
        "task_id",
        "language_template",
        "material_properties",
        "material_list",
        "objects",
        "min_object_gap_m",
        "target_index",
        "tray",
        "robot",
        "controller",
        "expert",
        "tactile",
        "temperature",
        "table",
        "recording",
        "cameras",
        "max_episode_steps",
        "success_criterion",
        "stable_steps",
        "tray_rest_tol_m",
        "seed",
    }
)

_SECTION_REQUIRED: dict[str, frozenset[str]] = {
    "robot": frozenset(
        {
            "urdf",
            "merge_fixed_links",
            "hand_link_name",
            "finger_link_names",
            "arm_joint_names",
            "finger_joint_names",
            "tcp_offset",
            "open_finger_pos",
            "closed_finger_pos",
            "base_pos",
            "base_quat",
            "init_qpos",
            "arm_kp",
            "arm_kv",
            "finger_kp",
            "finger_kv",
            "arm_force_min",
            "arm_force_max",
            "finger_force_min",
            "finger_force_max",
        }
    ),
    "controller": frozenset(
        {
            "pos_scale_m",
            "rot_scale_rad",
            "ik_damping",
            "ik_max_iters",
            "ik_max_samples",
            "control_hz",
            "sim_dt",
            "decimation",
            "finger_speed_m_per_s",
        }
    ),
    "expert": frozenset(
        {
            "approach",
            "temperature_judge_enabled",
            "temperature_candidate",
            "temperature_judge_threshold_c",
        }
    ),
    "tactile": frozenset({"modalities"}),
    "temperature": frozenset({"finger_grid", "object_grid", "ambient_temp_c"}),
    "table": frozenset({"pos", "size", "height_m"}),
    "recording": frozenset(
        {
            "output_dir",
            "keep_failed",
            "compression",
            "compression_opts",
            "contact_burst",
            "burst_pre_s",
            "burst_post_s",
            "burst_max_windows",
        }
    ),
    "cameras": frozenset(
        {
            "enabled",
            "resolution",
            "agentview_pos",
            "agentview_lookat",
            "fov",
            "eye_in_hand",
            "eye_in_hand_pos",
            "eye_in_hand_lookat",
            "eye_in_hand_up",
            "eye_in_hand_fov",
            "near",
            "far",
        }
    ),
}
_SECTION_OPTIONAL: dict[str, frozenset[str]] = {
    "table": frozenset({"color"}),
}

_OBJECT_COMMON = frozenset(
    {"shape", "mass_kg", "temp_c", "xy_jitter_m", "yaw_jitter_rad", "place_xy"}
)
_SHAPE_KEYS: dict[str, frozenset[str]] = {
    "cylinder": frozenset({"radius_m", "height_m"}),
    "box": frozenset({"radius_m", "height_m"}),
    "cube": frozenset({"side_m"}),
    "mesh": frozenset({"mesh_file", "mesh_scale", "mesh_euler_deg", "radius_m", "height_m"}),
}
_TACTILE_MODALITIES = frozenset({"force", "elastomer"})
_MATERIAL_PROPERTY_KEYS = frozenset(
    {"conductivity", "density", "specific_heat", "emissivity"}
)


# validation


def _check_keys(
    data: dict[str, Any],
    *,
    required: frozenset[str],
    optional: frozenset[str],
    where: str,
    err,
) -> None:
    missing = required - data.keys()
    unknown = data.keys() - required - optional
    if missing:
        err(f"{where}: missing key(s) {sorted(missing)}")
    if unknown:
        err(f"{where}: unknown key(s) {sorted(unknown)}")


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _num(value: Any, where: str) -> float:
    if not _is_num(value):
        raise ValueError(f"{where} must be a number, got {value!r}")
    return float(value)


def _int(value: Any, where: str, *, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{where} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{where} must be >= {minimum}, got {value!r}")
    return value


def _num_list(value: Any, where: str, length: int) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{where} must be a list of {length} numbers, got {value!r}")
    return [_num(v, f"{where}[{i}]") for i, v in enumerate(value)]


def _range(value: Any, where: str, *, positive: bool = True) -> list[float]:
    lo, hi = _num_list(value, where, 2)
    if lo > hi:
        raise ValueError(f"{where} must be [low, high] with low <= high, got {value!r}")
    if positive and lo <= 0.0:
        raise ValueError(f"{where} must be positive, got {value!r}")
    return [lo, hi]


def _str_list(value: Any, where: str, *, length: int | None = None) -> list[str]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{where} must be a non-empty list of strings, got {value!r}")
    if length is not None and len(value) != length:
        raise ValueError(f"{where} must have {length} entries, got {value!r}")
    if not all(isinstance(v, str) and v for v in value):
        raise ValueError(f"{where} must contain non-empty strings, got {value!r}")
    return list(value)


def _check_color(value: Any, where: str) -> None:
    if not isinstance(value, (list, tuple)) or len(value) not in (3, 4):
        raise ValueError(f"{where} must be an RGB(A) list of 3 or 4 numbers, got {value!r}")
    for i, v in enumerate(value):
        _num(v, f"{where}[{i}]")


def _validate(cfg: dict[str, Any], source: str) -> None:
    def err(message: str) -> None:
        raise ValueError(f"{source}: {message}")

    _check_keys(
        cfg,
        required=_TOP_REQUIRED,
        optional=frozenset(),
        where="top level",
        err=err,
    )

    if not isinstance(cfg["task_id"], str) or not cfg["task_id"]:
        err("task_id must be a non-empty string")
    _num(cfg["min_object_gap_m"], f"{source}: min_object_gap_m")
    _int(cfg["max_episode_steps"], f"{source}: max_episode_steps", minimum=1)
    _int(cfg["stable_steps"], f"{source}: stable_steps", minimum=1)
    _int(cfg["seed"], f"{source}: seed")
    if _num(cfg["tray_rest_tol_m"], f"{source}: tray_rest_tol_m") <= 0.0:
        err("tray_rest_tol_m must be positive")

    # --- sections ---------------------------------------------------------
    for name, required in _SECTION_REQUIRED.items():
        section = cfg[name]
        if not isinstance(section, dict):
            err(f"{name} must be a mapping")
        _check_keys(
            section,
            required=required,
            optional=_SECTION_OPTIONAL.get(name, frozenset()),
            where=name,
            err=err,
        )

    tray = cfg["tray"]
    if not isinstance(tray, dict):
        err("tray must be a mapping")
    _check_keys(
        tray,
        required=frozenset({"enabled", "place_xy"}),
        optional=frozenset(
            {"inner_x_m", "inner_y_m", "rim_m", "wall_m", "floor_m", "fuse_m", "color"}
        ),
        where="tray",
        err=err,
    )

    # --- objects ----------------------------------------------------------
    objects = cfg["objects"]
    if not isinstance(objects, list) or not objects:
        err("objects must be a non-empty list")
    for i, obj in enumerate(objects):
        where = f"objects[{i}]"
        if not isinstance(obj, dict):
            err(f"{where} must be a mapping")
        shape = obj.get("shape")
        if shape not in _SHAPE_KEYS:
            err(f"{where}.shape must be one of {sorted(_SHAPE_KEYS)}, got {shape!r}")
        _check_keys(
            obj,
            required=_OBJECT_COMMON | _SHAPE_KEYS[shape],
            optional=frozenset({"color"}),
            where=where,
            err=err,
        )
        _range(obj["mass_kg"], f"{source}: {where}.mass_kg")
        lo, hi = _num_list(obj["temp_c"], f"{source}: {where}.temp_c", 2)
        if lo > hi:
            err(f"{where}.temp_c must be [low, high] with low <= high")
        if _num(obj["xy_jitter_m"], f"{source}: {where}.xy_jitter_m") < 0.0:
            err(f"{where}.xy_jitter_m must be >= 0")
        if _num(obj["yaw_jitter_rad"], f"{source}: {where}.yaw_jitter_rad") < 0.0:
            err(f"{where}.yaw_jitter_rad must be >= 0")
        _num_list(obj["place_xy"], f"{source}: {where}.place_xy", 2)
        if "color" in obj:
            _check_color(obj["color"], f"{source}: {where}.color")
        if "radius_m" in obj:
            _range(obj["radius_m"], f"{source}: {where}.radius_m")
        if "height_m" in obj:
            _range(obj["height_m"], f"{source}: {where}.height_m")
        if "side_m" in obj:
            _range(obj["side_m"], f"{source}: {where}.side_m")
        if "mesh_file" in obj:
            if not isinstance(obj["mesh_file"], str) or not obj["mesh_file"]:
                err(f"{where}.mesh_file must be a non-empty string")
        if "mesh_scale" in obj:
            scale = obj["mesh_scale"]
            if _is_num(scale):
                if scale <= 0.0:
                    err(f"{where}.mesh_scale must be positive")
            else:
                for j, v in enumerate(_num_list(scale, f"{source}: {where}.mesh_scale", 3)):
                    if v <= 0.0:
                        err(f"{where}.mesh_scale[{j}] must be positive")
        if "mesh_euler_deg" in obj:
            _num_list(obj["mesh_euler_deg"], f"{source}: {where}.mesh_euler_deg", 3)

    # --- materials --------------------------------------------------------
    props = cfg["material_properties"]
    if not isinstance(props, dict) or not props:
        err("material_properties must be a non-empty mapping")
    for name, values in props.items():
        where = f"material_properties.{name}"
        if not isinstance(values, dict):
            err(f"{where} must be a mapping")
        _check_keys(
            values,
            required=_MATERIAL_PROPERTY_KEYS,
            optional=frozenset(),
            where=where,
            err=err,
        )
        for key in _MATERIAL_PROPERTY_KEYS:
            _num(values[key], f"{source}: {where}.{key}")

    material_list = _str_list(cfg["material_list"], f"{source}: material_list")
    if len(material_list) != len(objects):
        err(
            f"material_list has {len(material_list)} entries but objects has "
            f"{len(objects)}; one material per object"
        )
    for i, material in enumerate(material_list):
        if material not in props:
            err(f"material_list[{i}] = {material!r} is not in material_properties")

    # --- target -----------------------------------------------------------
    target = cfg["target_index"]
    if not isinstance(target, int) or isinstance(target, bool) or not (
        0 <= target < len(objects)
    ):
        err(f"target_index must be an integer in [0, {len(objects)}), got {target!r}")

    # --- language ---------------------------------------------------------
    template = cfg["language_template"]
    templates = [template] if isinstance(template, str) else template
    if not isinstance(templates, list) or not templates:
        err("language_template must be a string or a non-empty list of strings")
    if not all(isinstance(t, str) and t for t in templates):
        err("language_template entries must be non-empty strings")
    placeholder_sets = {frozenset(re.findall(r"\{(\w+)\}", t)) for t in templates}
    if len(placeholder_sets) > 1:
        err(f"language_template entries must use the same placeholders, got {[sorted(s) for s in placeholder_sets]}")
    unsupported = set().union(*placeholder_sets) - {"material", "shape"}
    if unsupported:
        err(f"language_template uses unsupported placeholder(s) {sorted(unsupported)}")

    # --- expert -----------------------------------------------------------
    expert = cfg["expert"]
    if expert["approach"] != "place_in_tray":
        err(f"expert.approach must be 'place_in_tray', got {expert['approach']!r}")
    judge = expert["temperature_judge_enabled"]
    if not isinstance(judge, bool):
        err("expert.temperature_judge_enabled must be a boolean")
    candidates = expert["temperature_candidate"]
    if not isinstance(candidates, (list, tuple)) or any(
        not isinstance(i, int) or isinstance(i, bool) for i in candidates
    ):
        err("expert.temperature_candidate must be a list of integers")
    if judge and not candidates:
        err("expert.temperature_judge_enabled needs temperature_candidate")
    if len(set(candidates)) != len(candidates) or any(
        i < 0 or i >= len(objects) for i in candidates
    ):
        err(
            f"expert.temperature_candidate must be distinct indices in [0, {len(objects)}), "
            f"got {list(candidates)}"
        )
    if judge and target not in candidates:
        # With the judge on, nothing is carried unless a touch matches the
        # interval, and only candidates are ever touched: a target outside the
        # list can never reach the tray, so every episode would be a failure.
        err(
            f"expert.temperature_candidate must include target_index={target!r} when the "
            f"temperature judge is enabled, got {list(candidates)}"
        )
    threshold = expert["temperature_judge_threshold_c"]
    if not isinstance(threshold, (list, tuple)) or len(threshold) != 2:
        err("expert.temperature_judge_threshold_c must be [low, high] (null = open end)")
    lo, hi = (None if v is None else _num(v, f"{source}: expert.temperature_judge_threshold_c") for v in threshold)
    if lo is None and hi is None:
        err("expert.temperature_judge_threshold_c needs at least one finite end")
    if lo is not None and hi is not None and lo > hi:
        err(f"expert.temperature_judge_threshold_c low must be <= high, got [{lo}, {hi}]")

    # --- tactile / temperature -------------------------------------------
    modalities = _str_list(cfg["tactile"]["modalities"], f"{source}: tactile.modalities")
    if len(set(modalities)) != len(modalities):
        err("tactile.modalities must not contain duplicates")
    unknown_modalities = set(modalities) - _TACTILE_MODALITIES
    if unknown_modalities:
        err(f"tactile.modalities supports {sorted(_TACTILE_MODALITIES)}, got {sorted(unknown_modalities)}")

    temperature = cfg["temperature"]
    for key in ("finger_grid", "object_grid"):
        grid = temperature[key]
        if not isinstance(grid, (list, tuple)) or len(grid) != 3:
            err(f"temperature.{key} must be a list of 3 positive integers, got {grid!r}")
        for i, v in enumerate(grid):
            _int(v, f"{source}: temperature.{key}[{i}]", minimum=1)
    _num(temperature["ambient_temp_c"], f"{source}: temperature.ambient_temp_c")

    # --- table / tray -----------------------------------------------------
    table = cfg["table"]
    _num_list(table["pos"], f"{source}: table.pos", 3)
    size = _num_list(table["size"], f"{source}: table.size", 3)
    if any(v <= 0.0 for v in size):
        err("table.size entries must be positive")
    _num(table["height_m"], f"{source}: table.height_m")
    if "color" in table:
        _check_color(table["color"], f"{source}: table.color")

    if not isinstance(tray["enabled"], bool):
        err("tray.enabled must be a boolean")
    _num_list(tray["place_xy"], f"{source}: tray.place_xy", 2)
    tray_geometry = ("inner_x_m", "inner_y_m", "rim_m", "wall_m", "floor_m", "fuse_m", "color")
    if tray["enabled"]:
        missing = [k for k in tray_geometry if k not in tray]
        if missing:
            err(f"tray: missing key(s) {missing} (required when tray.enabled is true)")
        for key in ("inner_x_m", "inner_y_m", "rim_m", "wall_m", "floor_m", "fuse_m"):
            if _num(tray[key], f"{source}: tray.{key}") <= 0.0:
                err(f"tray.{key} must be positive")
        _check_color(tray["color"], f"{source}: tray.color")

    # --- success ----------------------------------------------------------
    if cfg["success_criterion"] != "place_in_tray":
        err(f"success_criterion must be 'place_in_tray', got {cfg['success_criterion']!r}")
    if not tray["enabled"]:
        err("success_criterion 'place_in_tray' needs tray.enabled: true")

    # --- recording / cameras ---------------------------------------------
    recording = cfg["recording"]
    if not isinstance(recording["output_dir"], str) or not recording["output_dir"]:
        err("recording.output_dir must be a non-empty string")
    if not isinstance(recording["compression"], str) or not recording["compression"]:
        err("recording.compression must be a non-empty string")
    _int(recording["compression_opts"], f"{source}: recording.compression_opts")
    _int(recording["burst_max_windows"], f"{source}: recording.burst_max_windows", minimum=0)
    for key in ("burst_pre_s", "burst_post_s"):
        if _num(recording[key], f"{source}: recording.{key}") < 0.0:
            err(f"recording.{key} must be >= 0")
    for key in ("keep_failed", "contact_burst"):
        if not isinstance(recording[key], bool):
            err(f"recording.{key} must be a boolean")

    cameras = cfg["cameras"]
    if not isinstance(cameras["enabled"], bool):
        err("cameras.enabled must be a boolean")
    if not isinstance(cameras["eye_in_hand"], bool):
        err("cameras.eye_in_hand must be a boolean")
    resolution = cameras["resolution"]
    if not isinstance(resolution, (list, tuple)) or len(resolution) != 2:
        err("cameras.resolution must be [width, height]")
    for i, v in enumerate(resolution):
        _int(v, f"{source}: cameras.resolution[{i}]", minimum=1)
    for key in ("agentview_pos", "agentview_lookat", "eye_in_hand_pos", "eye_in_hand_lookat", "eye_in_hand_up"):
        _num_list(cameras[key], f"{source}: cameras.{key}", 3)
    for key in ("fov", "eye_in_hand_fov", "near", "far"):
        _num(cameras[key], f"{source}: cameras.{key}")

    # --- robot / controller ----------------------------------------------
    robot = cfg["robot"]
    if not isinstance(robot["urdf"], str) or not robot["urdf"]:
        err("robot.urdf must be a non-empty string")
    if not isinstance(robot["merge_fixed_links"], bool):
        err("robot.merge_fixed_links must be a boolean")
    for key in ("hand_link_name",):
        if not isinstance(robot[key], str) or not robot[key]:
            err(f"robot.{key} must be a non-empty string")
    _str_list(robot["finger_link_names"], f"{source}: robot.finger_link_names", length=2)
    arm_names = _str_list(robot["arm_joint_names"], f"{source}: robot.arm_joint_names")
    finger_names = _str_list(robot["finger_joint_names"], f"{source}: robot.finger_joint_names", length=2)
    _num_list(robot["tcp_offset"], f"{source}: robot.tcp_offset", 3)
    _num(robot["open_finger_pos"], f"{source}: robot.open_finger_pos")
    _num(robot["closed_finger_pos"], f"{source}: robot.closed_finger_pos")
    _num_list(robot["base_pos"], f"{source}: robot.base_pos", 3)
    _num_list(robot["base_quat"], f"{source}: robot.base_quat", 4)
    qpos = robot["init_qpos"]
    if not isinstance(qpos, (list, tuple)) or len(qpos) != len(arm_names) + len(finger_names):
        err(f"robot.init_qpos must have {len(arm_names) + len(finger_names)} entries")
    for i, v in enumerate(qpos):
        _num(v, f"{source}: robot.init_qpos[{i}]")
    for key in ("arm_kp", "arm_kv", "arm_force_min", "arm_force_max"):
        values = _num_list(robot[key], f"{source}: robot.{key}", len(arm_names))
        if key in ("arm_kp", "arm_kv") and any(v < 0.0 for v in values):
            err(f"robot.{key} must be >= 0")
    for key in ("finger_kp", "finger_kv", "finger_force_min", "finger_force_max"):
        values = _num_list(robot[key], f"{source}: robot.{key}", len(finger_names))
        if key in ("finger_kp", "finger_kv") and any(v < 0.0 for v in values):
            err(f"robot.{key} must be >= 0")

    controller = cfg["controller"]
    for key in ("pos_scale_m", "rot_scale_rad", "ik_damping", "control_hz", "sim_dt"):
        if _num(controller[key], f"{source}: controller.{key}") <= 0.0:
            err(f"controller.{key} must be positive")
    _int(controller["ik_max_iters"], f"{source}: controller.ik_max_iters", minimum=1)
    _int(controller["ik_max_samples"], f"{source}: controller.ik_max_samples", minimum=1)
    _int(controller["decimation"], f"{source}: controller.decimation", minimum=1)
    if _num(controller["finger_speed_m_per_s"], f"{source}: controller.finger_speed_m_per_s") < 0.0:
        err("controller.finger_speed_m_per_s must be >= 0")


# --------------------------------------------------------------------- helpers


def language_templates(cfg: TaskConfig) -> tuple[str, ...]:
    """Every instruction phrasing; a single template is a one-tuple."""
    template = cfg["language_template"]
    return (template,) if isinstance(template, str) else tuple(template)


def tray_floor_z(tray: dict[str, Any], table_height_m: float) -> float:
    """World z of the cavity floor's top face — the surface a body rests on."""
    return float(table_height_m) + float(tray["floor_m"]) - float(tray["fuse_m"])


def tray_rim_z(tray: dict[str, Any], table_height_m: float) -> float:
    """World z of the walls' top edge."""
    return tray_floor_z(tray, table_height_m) + float(tray["rim_m"])


# ---------------------------------------------------------------------- load


def load_task_config(path: str | Path) -> TaskConfig:
    """Load one task YAML into a validated plain mapping."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Task config must be a YAML mapping: {path}")
    _validate(raw, str(path))
    return raw

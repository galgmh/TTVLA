"""Load the scene/object/sensor/task configuration hierarchy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TaskConfig = dict[str, Any]
DEFAULT_SCENE_CONFIG = "sim/configs/scene/cup_on_table.yaml"


def project_path(path: str | Path) -> Path:
    """Resolve a path relative to the repository root."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = project_path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return raw


def load_object_config(path: str | Path) -> dict[str, Any]:
    cfg = load_yaml(path)
    manifest_path = project_path(cfg["manifest"])
    manifest = load_yaml(manifest_path)

    cfg = dict(cfg)
    cfg["manifest_path"] = str(manifest_path)
    cfg["urdf_path"] = str(manifest_path.parent / manifest["urdf_file"])
    cfg["asset_version"] = manifest["asset_version"]
    cfg["region_order"] = list(manifest["region_order"])
    cfg["region_links"] = {
        name: manifest["regions"][name]["link"] for name in cfg["region_order"]
    }
    cfg["region_z_spans"] = {
        name: list(manifest["regions"][name]["z_span_m"]) for name in cfg["region_order"]
    }
    cfg["link_names"] = list(manifest["link_names"])
    cfg["base_euler_deg"] = list(manifest["base_euler_deg"])
    cfg["mass_kg"] = float(manifest["mass_kg"])
    cfg["aabb_min_m"] = list(manifest["aabb_min_m"])
    cfg["aabb_max_m"] = list(manifest["aabb_max_m"])
    return cfg


def load_sensor_config(path: str | Path) -> dict[str, Any]:
    return load_yaml(path)


def _legacy_cup_view(cup: dict[str, Any], support: dict[str, Any]) -> dict[str, Any]:
    """Expose the existing simulator keys while the runtime is refactored.

    The YAML source remains clean and nested.  This adapter prevents scene and
    recording code from depending on task-specific trajectory configuration.
    """
    placement = dict(cup["placement"])
    thermal = dict(cup["thermal"])
    support_thermal = dict(support["thermal"])
    materials = dict(thermal["materials"])
    materials.update(dict(support_thermal.get("materials", {})))
    return {
        **cup,
        "place_xy": list(placement["xy"]),
        "rest_height_m": float(support["geometry"]["size"][2]),
        "xy_jitter_m": float(placement.get("xy_jitter_m", 0.0)),
        "yaw_jitter_rad": float(placement.get("yaw_jitter_rad", 0.0)),
        "color": list(cup.get("appearance", {}).get("color", [0.85, 0.84, 0.80])),
        "material": str(thermal["default_material"]),
        "link_materials": dict(thermal["link_materials"]),
        "material_properties": materials,
    }


def load_scene_config(path: str | Path = DEFAULT_SCENE_CONFIG) -> TaskConfig:
    """Load one reusable scene and expand its object/sensor references."""
    scene_path = project_path(path)
    scene = dict(load_yaml(scene_path))
    object_refs = dict(scene.pop("objects"))
    cup_path = project_path(object_refs["cup"])
    support_path = project_path(object_refs["support"])
    sensor_path = project_path(scene.pop("sensor"))

    cup = load_object_config(cup_path)
    support = load_yaml(support_path)
    cup_runtime = _legacy_cup_view(cup, support)
    support_size = list(support["geometry"]["size"])
    support_placement = dict(support["placement"])

    scene["object"] = cup_runtime
    scene["objects"] = {"cup": cup, "support": support}
    scene["sensor"] = load_sensor_config(sensor_path)
    scene["platform"] = {
        "size_xy": support_size[:2],
        "height_m": float(support_size[2]),
        "place_xy": list(support_placement["xy"]),
        "z_above_table_m": float(support_placement.get("z_above_table_m", 0.0)),
        "material": str(support["thermal"]["material"]),
        "color": list(support["appearance"]["color"]),
    }
    scene["config_paths"] = {
        "scene": str(scene_path),
        "cup": str(cup_path),
        "support": str(support_path),
        "sensor": str(sensor_path),
    }
    return scene


def load_task_config(
    path: str | Path,
    scene_path: str | Path = DEFAULT_SCENE_CONFIG,
) -> TaskConfig:
    """Combine a goal-only task with a reusable scene.

    A task YAML is intentionally limited to ``task_id``, ``instructions`` and
    ``success``.  Scripted expert geometry is code-owned and injected only as a
    temporary compatibility view for the existing controllers.
    """
    from sim.control.cup_expert_defaults import (
        EXPERT_CONTACT,
        EXPERT_REGIONS,
        EXPERT_TIMING,
        MIXED_PROBE,
    )

    task_path = project_path(path)
    task = load_yaml(task_path)
    expected = {"task_id", "instructions", "success"}
    if set(task) != expected:
        raise ValueError(
            f"{task_path} 只能包含 {sorted(expected)}，实为 {sorted(task)}"
        )
    cfg = load_scene_config(scene_path)
    cfg.update(task)
    instructions = cfg["instructions"]
    if not isinstance(instructions, list) or not instructions:
        raise ValueError(f"{task_path} instructions 必须是非空列表")
    if any(not isinstance(value, str) or not value.strip() for value in instructions):
        raise ValueError(f"{task_path} instructions 中每一项都必须是非空字符串")
    if len(set(instructions)) != len(instructions):
        raise ValueError(f"{task_path} instructions 不能包含重复项")
    cfg["regions"] = EXPERT_REGIONS
    cfg["mixed_probe"] = MIXED_PROBE
    cfg["timing"] = EXPERT_TIMING
    cfg["contact"] = EXPERT_CONTACT
    cfg["config_paths"] = {**cfg["config_paths"], "task": str(task_path)}
    return cfg


def load_dataset_config(path: str | Path) -> dict[str, Any]:
    """Load a generation/data preprocessing preset (not a scene config)."""
    return load_yaml(path)


__all__ = [
    "REPO_ROOT",
    "TaskConfig",
    "DEFAULT_SCENE_CONFIG",
    "load_dataset_config",
    "load_object_config",
    "load_sensor_config",
    "load_scene_config",
    "load_task_config",
    "load_yaml",
    "project_path",
]

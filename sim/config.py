"""Small YAML helpers for the Genesis task and dataset configs.

The project keeps configuration intentionally lightweight. We only need to
load mappings, resolve project-relative paths, and expand the object manifest.
The simulator and models validate their own required fields when they run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TaskConfig = dict[str, Any]


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


def load_task_config(path: str | Path) -> TaskConfig:
    task_path = project_path(path)
    cfg = load_yaml(task_path)

    object_path = project_path(cfg["object"])
    sensor_path = project_path(cfg["sensor"])
    cfg = dict(cfg)
    cfg["object"] = load_object_config(object_path)
    cfg["sensor"] = load_sensor_config(sensor_path)
    cfg["config_paths"] = {
        "task": str(task_path),
        "object": str(object_path),
        "sensor": str(sensor_path),
    }
    return cfg


def load_dataset_config(path: str | Path) -> dict[str, Any]:
    return load_yaml(path)


__all__ = [
    "REPO_ROOT",
    "TaskConfig",
    "load_dataset_config",
    "load_object_config",
    "load_sensor_config",
    "load_task_config",
    "load_yaml",
    "project_path",
]

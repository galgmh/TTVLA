"""Inspect robot joint-position limits across one or more HDF5 datasets."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np

from sim.config import load_yaml, project_path
from sim.recording import read_metadata


@dataclass
class JointLimit:
    name: str
    lower: float
    upper: float


@dataclass
class JointStats:
    limit: JointLimit
    minimum: float = math.inf
    maximum: float = -math.inf
    lower_margin: float = math.inf
    upper_margin: float = math.inf
    near_frames: int = 0
    violations: int = 0
    worst_episode: str = ""
    worst_frame: int = -1

    def update(
        self,
        values: np.ndarray,
        *,
        episode: str,
        threshold: float,
    ) -> None:
        lower_margin = values - self.limit.lower
        upper_margin = self.limit.upper - values
        margins = np.minimum(lower_margin, upper_margin)
        frame = int(np.argmin(margins))
        if float(margins[frame]) < min(self.lower_margin, self.upper_margin):
            self.worst_episode = episode
            self.worst_frame = frame

        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))
        self.lower_margin = min(self.lower_margin, float(lower_margin.min()))
        self.upper_margin = min(self.upper_margin, float(upper_margin.min()))
        self.near_frames += int(
            np.count_nonzero((lower_margin <= threshold) | (upper_margin <= threshold))
        )
        self.violations += int(
            np.count_nonzero((values < self.limit.lower - 1e-8) | (values > self.limit.upper + 1e-8))
        )


def _files_from_sources(sources: list[str]) -> list[Path]:
    files: set[Path] = set()
    for source in sources:
        path = project_path(source)
        if path.is_file() and path.suffix == ".hdf5":
            files.add(path.resolve())
        elif path.is_dir():
            files.update(p.resolve() for p in path.rglob("*.hdf5"))
        else:
            raise SystemExit(f"source is not an HDF5 file or directory: {path}")
    return sorted(files)


def _limits_from_scene(scene_path: str | Path) -> list[JointLimit]:
    scene = load_yaml(scene_path)
    robot = dict(scene["robot"])
    urdf_path = project_path(robot["urdf"])
    root = ET.parse(urdf_path).getroot()
    by_name = {joint.attrib["name"]: joint for joint in root.findall("joint")}

    names = list(robot["arm_joint_names"]) + list(robot["finger_joint_names"])
    limits: list[JointLimit] = []
    for name in names:
        joint = by_name.get(str(name))
        if joint is None:
            raise ValueError(f"URDF {urdf_path} does not contain joint {name!r}")
        limit = joint.find("limit")
        if limit is None or "lower" not in limit.attrib or "upper" not in limit.attrib:
            raise ValueError(f"joint {name!r} has no bounded lower/upper limits")
        limits.append(
            JointLimit(
                name=str(name),
                lower=float(limit.attrib["lower"]),
                upper=float(limit.attrib["upper"]),
            )
        )
    return limits


def _is_finger_joint(name: str) -> bool:
    return "finger" in name


def _format_value(name: str, value: float) -> str:
    if _is_finger_joint(name):
        return f"{value * 1000.0:8.3f}mm"
    return f"{math.degrees(value):8.2f}°"


def inspect(
    files: list[Path],
    *,
    arm_margin_rad: float,
    finger_margin_m: float,
) -> tuple[dict[tuple[str, str], JointStats], list[str]]:
    scene_limits: dict[str, list[JointLimit]] = {}
    stats: dict[tuple[str, str], JointStats] = {}
    issues: list[str] = []

    for file_index, path in enumerate(files, start=1):
        try:
            with h5py.File(path, "r") as handle:
                metadata = read_metadata(handle)
                positions = np.asarray(handle["robot/joint_position"][...], dtype=np.float64)
        except Exception as exc:
            issues.append(f"{path}: cannot read robot/joint_position ({exc})")
            continue

        scene_path = str(metadata.get("config_paths", {}).get("scene", ""))
        if not scene_path:
            issues.append(f"{path}: metadata has no config_paths.scene")
            continue
        if scene_path not in scene_limits:
            scene_limits[scene_path] = _limits_from_scene(scene_path)
        limits = scene_limits[scene_path]
        if positions.ndim != 2 or positions.shape[1] != len(limits):
            issues.append(
                f"{path}: joint_position shape {positions.shape}, expected (*,{len(limits)})"
            )
            continue

        for index, limit in enumerate(limits):
            threshold = (
                finger_margin_m if _is_finger_joint(limit.name) else arm_margin_rad
            )
            key = (scene_path, limit.name)
            stat = stats.setdefault(key, JointStats(limit=limit))
            stat.update(
                positions[:, index],
                episode=str(metadata.get("episode_id", path.stem)),
                threshold=threshold,
            )

        if file_index % 20 == 0 or file_index == len(files):
            print(f"checked {file_index}/{len(files)} episodes")
    return stats, issues


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", help="HDF5 files or directories")
    parser.add_argument(
        "--arm-margin-rad",
        type=float,
        default=0.10,
        help="report frames within this many radians of an arm limit",
    )
    parser.add_argument(
        "--finger-margin-m",
        type=float,
        default=0.002,
        help="report frames within this many metres of a finger limit",
    )
    parser.add_argument("--json", default=None, help="optional JSON report path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    files = _files_from_sources(args.sources)
    if not files:
        raise SystemExit("no HDF5 files found")

    stats, issues = inspect(
        files,
        arm_margin_rad=float(args.arm_margin_rad),
        finger_margin_m=float(args.finger_margin_m),
    )
    report = []
    print("\nscene | joint | min | max | lower | upper | lower margin | upper margin | near frames")
    for (scene_path, name), stat in sorted(stats.items()):
        limit = stat.limit
        print(
            f"{Path(scene_path).name} | {name} | "
            f"{_format_value(name, stat.minimum)} | "
            f"{_format_value(name, stat.maximum)} | "
            f"{_format_value(name, limit.lower)} | "
            f"{_format_value(name, limit.upper)} | "
            f"{_format_value(name, stat.lower_margin)} | "
            f"{_format_value(name, stat.upper_margin)} | "
            f"{stat.near_frames}"
        )
        report.append(
            {
                "scene": scene_path,
                "joint": name,
                "minimum": stat.minimum,
                "maximum": stat.maximum,
                "lower": limit.lower,
                "upper": limit.upper,
                "lower_margin": stat.lower_margin,
                "upper_margin": stat.upper_margin,
                "near_frames": stat.near_frames,
                "violations": stat.violations,
                "worst_episode": stat.worst_episode,
                "worst_frame": stat.worst_frame,
            }
        )

    for issue in issues:
        print(f"ERROR: {issue}")
    violations = sum(stat.violations for stat in stats.values())
    print(f"\nepisodes={len(files)} violations={violations} read_errors={len(issues)}")

    if args.json:
        output = Path(args.json).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "episodes": len(files),
                    "arm_margin_rad": float(args.arm_margin_rad),
                    "finger_margin_m": float(args.finger_margin_m),
                    "joints": report,
                    "read_errors": issues,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    return 1 if issues or violations else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Deterministic episode sampling for the direct lift tasks."""

from __future__ import annotations

import zlib
from typing import Any

import numpy as np

from sim.episode import EpisodeSpec

SPLITS = ("train", "validation", "test")


def _split_code(split: str) -> int:
    return int(zlib.crc32(split.encode("utf-8")))


def _rotate_about_axis(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    return (
        vector * np.cos(angle)
        + np.cross(axis, vector) * np.sin(angle)
        + axis * float(axis @ vector) * (1.0 - np.cos(angle))
    )


def sample_initial_link_temperatures(
    task_cfg: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[float, ...]:
    """Sample one initial temperature per cup region from the object config."""
    region_order = tuple(str(v) for v in task_cfg["object"]["region_order"])
    ranges = task_cfg["object"]["thermal"]["initial_region_temperature_ranges_c"]
    if set(ranges) != set(region_order):
        raise ValueError(
            "initial_region_temperature_ranges_c must match object.region_order"
        )
    return tuple(float(rng.uniform(*ranges[name])) for name in region_order)


class LiftSampler:
    """Sample a reproducible lift episode from scene and object configuration."""

    def __init__(
        self,
        task_cfg: dict[str, Any],
        dataset_cfg: dict[str, Any],
        split: str,
        *,
        seed: int | None = None,
        count: int = 1,
        region: int | None = None,
    ):
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}; available: {SPLITS}")
        self.task = task_cfg
        self.dataset = dataset_cfg
        self.split = split
        self.seed = int(dataset_cfg["seed"] if seed is None else seed)
        self.count = int(count)
        self.region = None if region is None else int(region)
        self.region_order = tuple(str(v) for v in task_cfg["object"]["region_order"])
        if self.region is not None and not 0 <= self.region < len(self.region_order):
            raise ValueError(f"region index {self.region} is outside [0, 4)")

        self.material = dict(
            task_cfg["object"]["material_properties"][task_cfg["object"]["material"]]
        )
        self.instruction = str(
            task_cfg.get("instruction") or task_cfg.get("instructions", [""])[0]
        )
        self.z_spans = {
            name: tuple(float(v) for v in span)
            for name, span in task_cfg["object"]["region_z_spans"].items()
        }

    def _sample_probe(self, rng: np.random.Generator, region: int):
        name = self.region_order[region]
        cfg = self.task["regions"][name]
        xyz = np.asarray(cfg["pos"], dtype=np.float64).copy()
        xy_jitter = float(cfg.get("xy_jitter_m", 0.0))
        xyz[:2] += rng.uniform(-xy_jitter, xy_jitter, size=2)
        z_jitter = float(cfg.get("z_jitter_m", 0.0))
        xyz[2] += float(rng.uniform(-z_jitter, z_jitter))

        azimuth = 0.0
        if name != "handle":
            limit = float(cfg.get("azimuth_jitter_rad", 0.0))
            azimuth = float(rng.uniform(-limit, limit))
        c, s = np.cos(azimuth), np.sin(azimuth)
        rz = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        tool = rz @ np.asarray(cfg["tool_dir"], dtype=np.float64)
        finger = rz @ np.asarray(cfg["finger_dir"], dtype=np.float64)
        tilt = float(
            rng.uniform(
                -float(cfg.get("tilt_jitter_rad", 0.0)),
                float(cfg.get("tilt_jitter_rad", 0.0)),
            )
        )
        tool = _rotate_about_axis(tool, finger, tilt)
        tool /= np.linalg.norm(tool)
        finger -= float(finger @ tool) * tool
        finger /= np.linalg.norm(finger)

        lo, hi = self.z_spans[name]
        if not lo <= xyz[2] <= hi:
            raise ValueError(f"probe z={xyz[2]:.4f} outside {name} span [{lo}, {hi}]")
        return tuple(float(v) for v in xyz), tuple(float(v) for v in tool), tuple(
            float(v) for v in finger
        )

    def sample(self, index: int) -> EpisodeSpec:
        if not 0 <= index < self.count:
            raise IndexError(f"index {index} outside sampled range {self.count}")
        seq = np.random.SeedSequence([self.seed, _split_code(self.split), int(index)])
        rng = np.random.default_rng(seq)
        random_seed = int(seq.generate_state(1)[0])

        region = self.region if self.region is not None else int(rng.integers(0, 4))
        temperatures = sample_initial_link_temperatures(self.task, rng)
        position, tool_dir, finger_dir = self._sample_probe(rng, region)

        sensor_cfg = self.dataset["sensor_randomization"]
        nominal = self.task["sensor"]["nominal"]
        rc = float(nominal["rc_s"]) * float(rng.uniform(*sensor_cfg["rc_multiplier_range"]))
        noise = float(nominal["noise_c"]) * float(
            rng.uniform(*sensor_cfg["noise_multiplier_range"])
        )
        quantization = float(nominal["quantization_c"]) * float(
            rng.uniform(*sensor_cfg["quantization_multiplier_range"])
        )
        bias = float(rng.uniform(*sensor_cfg["bias_absolute_range"]))

        object_cfg = self.task["object"]
        jitter_xy = float(object_cfg["xy_jitter_m"])
        cup_xy = (
            float(rng.uniform(-jitter_xy, jitter_xy)),
            float(rng.uniform(-jitter_xy, jitter_xy)),
        )
        cup_yaw = float(
            rng.uniform(
                -float(object_cfg["yaw_jitter_rad"]),
                float(object_cfg["yaw_jitter_rad"]),
            )
        )
        ambient = float(self.task["sensor"]["ambient_temp_c"])
        return EpisodeSpec(
            episode_id=f"{self.dataset['dataset_version']}_{self.split}_{index:06d}",
            dataset_version=str(self.dataset["dataset_version"]),
            split=self.split,
            random_seed=random_seed,
            region_order=self.region_order,
            target_region=region,
            mixed=False,
            regime="scenario_1",
            finger_mode="ambient",
            link_temps_c=temperatures,
            fingertip_temps_c=(ambient, ambient),
            contact_quality="stable",
            contact_hold_s=float(self.task["timing"]["contact_s"]),
            sensor_rc_s=rc,
            sensor_noise_c=noise,
            sensor_bias_c=bias,
            sensor_quantization_c=quantization,
            probe_pos=position,
            probe_tool_dir=tool_dir,
            probe_finger_dir=finger_dir,
            probe_pre_dist=float(self.task["regions"][self.region_order[region]]["pre_dist"]),
            cup_xy_jitter_m=cup_xy,
            cup_yaw_jitter_rad=cup_yaw,
            material_parameters=tuple(sorted((str(k), float(v)) for k, v in self.material.items())),
            instruction=self.instruction,
        )

    def __iter__(self):
        for index in range(self.count):
            yield self.sample(index)


__all__ = ["LiftSampler", "SPLITS", "sample_initial_link_temperatures"]

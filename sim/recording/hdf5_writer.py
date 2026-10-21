"""Streaming LIBERO-style HDF5 writer (robomimic superset).

Layout (docs/architecture.md §4):

    <task_id>_<material>.hdf5
      attrs: format_version, task_id, generator
      data/
        attrs: env_args(JSON), problem_info(JSON), total
        demo_i/
          attrs: num_samples, success, episode_seed, episode_steps,
                 num_contact_events, num_windows, object_* ...  (runner._attrs)
          obs/<...>          # includes tactile/ and temperature/ subtrees
          events/<...>       # contact_events/, burst/ (optional)
          states (T,S)       # optional; not written in v1 (see architecture §5.8)
          actions (T,7), rewards (T,), dones (T,)
      layouts/<...>          # static probe geometry (per file, episode-independent)

One episode lives in memory at a time and is flushed to disk as soon as it
finishes, so generation RAM stays flat regardless of dataset size.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

# 1.1: per-modality probe grids + obs/link_pose
# 1.2: multi-object scenes: obs/object_pose + obs/temperature/objects_grid,
#      per-object demo attrs (objects_*, target_index, language_instruction),
#      env_args.objects. Single-object files remain byte-identical apart from
#      the version string.
# 1.3: optional obs/tactile/height_map (F,H,W) metres — the marker field
#      resampled onto a dense image (tactile.height_map.enabled, docs §5.2).
#      The tactile modality set is config-driven either way; a config without
#      the channel writes the same bytes as under 1.2.
# 1.5: official sensor model, measured-only recording (docs §9-R36). The finger
#      sensors are the engine's default lumped grid_size (1,1,1) — one measured
#      temperature per finger (T, n_fingers, 1, 1, 1) — with the official
#      non-idealities (sensor_time_constant RC lag, noise, resolution
#      quantization) applied to the measured channel; the object grids carry no
#      non-idealities and their read() equals the ground truth. temperature.
#      record_ground_truth: false (the dataset records read(), not the raw
#      field); layouts.attrs["temperature_sensor"] describes the configuration.
# 1.6: tactile reduced to two derived signals, both in the finger/link frame:
#      obs/tactile/force (T, n_fingers, 6) = [Fn(3), Ft(3)] per finger (AT-VLA
#      6D force, summed from the 12 probe forces), and obs/tactile/elastomer
#      (T, n_fingers*P, 2) = 2D marker displacement in the pad tangent plane.
#      obs/tactile/depth and obs/tactile/height_map are no longer written by the
#      new path; layouts.attrs["channel_formats"] declares both channels.
FORMAT_VERSION = "tempvla-1.6"

# Datasets larger than this (bytes) get compressed; smaller ones are cheap
# either way and stay uncompressed for fast random access.
_COMPRESS_THRESHOLD = 8192


@dataclass
class EpisodeData:
    """One finished episode, ready to be written as a demo_N group."""

    obs: dict[str, np.ndarray]
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    states: np.ndarray | None = None
    events: dict[str, np.ndarray] | None = None
    attrs: dict[str, Any] = field(default_factory=dict)


def _create_dataset(group: h5py.Group, name: str, array: Any, *, compression: str, compression_opts: int) -> None:
    arr = np.asarray(array)
    if arr.dtype.kind == "U":  # h5py has no unicode dtype; store as UTF-8 bytes
        arr = np.char.encode(arr, "utf-8")
    kwargs: dict[str, Any] = {}
    if compression and arr.ndim > 0 and arr.nbytes > _COMPRESS_THRESHOLD:
        kwargs.update(compression=compression, compression_opts=compression_opts, shuffle=True)
    group.create_dataset(name, data=arr, **kwargs)


def _write_nested(group: h5py.Group, mapping: dict[str, Any], create) -> None:
    """Write a {key: array-or-dict} mapping, dict values become nested groups.

    A nested ``"attrs"`` mapping sets group *attributes* rather than creating a
    subgroup of that name — that is how ``layouts/grids/<tag>/`` carries its
    ``probe_grid`` / ``modalities`` metadata without a parallel name space. No
    obs/events payload uses that key.
    """
    for key, value in mapping.items():
        if key == "attrs" and isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, np.ndarray) and v.dtype.kind == "U":
                    # h5py has no unicode dtype, for attributes either — a numpy
                    # '<U5' array raises "No conversion path for dtype" and takes
                    # the whole file down. Same encoding as _write_dataset.
                    v = np.char.encode(v, "utf-8")
                group.attrs[k] = v
            continue
        if isinstance(value, dict):
            _write_nested(group.create_group(key), value, create)
        else:
            create(group, key, value)


class _Target:
    """One open HDF5 file receiving demo_N groups."""

    def __init__(
        self,
        path: Path,
        *,
        task_id: str,
        problem_info: dict[str, Any],
        env_args: dict[str, Any],
        layouts: dict[str, Any] | None,
        compression: str,
        compression_opts: int,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(path, "w")
        self.file.attrs["format_version"] = FORMAT_VERSION
        self.file.attrs["task_id"] = task_id
        self.file.attrs["generator"] = json.dumps(
            {
                "project": "ThermTacVLA",
                "format_version": FORMAT_VERSION,
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )
        self.compression = compression
        self.compression_opts = compression_opts

        data = self.file.create_group("data")
        data.attrs["env_args"] = json.dumps(env_args, ensure_ascii=False)
        data.attrs["problem_info"] = json.dumps(problem_info, ensure_ascii=False)
        data.attrs["total"] = 0
        self.data = data
        self.n = 0

        if layouts:
            g = self.file.create_group("layouts")
            attrs = layouts.get("attrs")
            if attrs:
                for k, v in attrs.items():
                    g.attrs[k] = v
            _write_nested(g, {k: v for k, v in layouts.items() if k != "attrs"}, self._create)

    def _create(self, group: h5py.Group, name: str, array: Any) -> None:
        _create_dataset(group, name, array, compression=self.compression, compression_opts=self.compression_opts)

    def add_episode(self, ep: EpisodeData) -> str:
        name = f"demo_{self.n}"
        g = self.data.create_group(name)
        for k, v in ep.attrs.items():
            g.attrs[k] = v
        g.attrs["num_samples"] = int(len(ep.actions))

        _write_nested(g.create_group("obs"), ep.obs, self._create)
        self._create(g, "actions", np.asarray(ep.actions, dtype=np.float32))
        self._create(g, "rewards", np.asarray(ep.rewards, dtype=np.float32))
        self._create(g, "dones", np.asarray(ep.dones, dtype=bool))
        if ep.states is not None:
            self._create(g, "states", np.asarray(ep.states, dtype=np.float32))
        if ep.events:
            _write_nested(g.create_group("events"), ep.events, self._create)

        self.n += 1
        self.data.attrs["total"] = self.n
        self.file.flush()
        return name


class HDF5Writer:
    """Routes successful episodes to the main file, failures to a sibling file."""

    def __init__(
        self,
        path: str | Path,
        *,
        task_id: str,
        problem_info: dict[str, Any],
        env_args: dict[str, Any],
        layouts: dict[str, Any] | None = None,
        failed_path: str | Path | None = None,
        compression: str = "gzip",
        compression_opts: int = 4,
    ) -> None:
        common: dict[str, Any] = dict(
            task_id=task_id,
            problem_info=problem_info,
            env_args=env_args,
            layouts=layouts,
            compression=compression,
            compression_opts=compression_opts,
        )
        self._closed = False
        self._main = _Target(Path(path), **common)
        # Failed episodes get the same layouts/: they are tiny and make the
        # failed file self-contained for debugging.
        self._failed = _Target(Path(failed_path), **common) if failed_path else None

    def add_episode(self, ep: EpisodeData, *, success: bool) -> str:
        target = self._main if (success or self._failed is None) else self._failed
        return target.add_episode(ep)

    @property
    def num_episodes(self) -> int:
        return self._main.n

    @property
    def num_failed(self) -> int:
        return self._failed.n if self._failed is not None else 0

    def close(self) -> None:
        if self._closed:  # idempotent: generate.py closes in `finally`, users may also `with`
            return
        self._closed = True
        self._main.file.close()
        if self._failed is not None:
            # The failed file is opened eagerly (so a mid-run crash still leaves a
            # valid, self-contained file). With zero failures it would otherwise
            # stay behind as an empty demo-less HDF5 that `data/*.hdf5` globs pick
            # up and validate_dataset.py flags as "no demo_N groups found".
            path = self._failed.file.filename
            empty = self._failed.n == 0
            self._failed.file.close()
            if empty:
                Path(path).unlink(missing_ok=True)

    def __enter__(self) -> "HDF5Writer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

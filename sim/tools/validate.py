"""Dataset validation against the ThermTacVLA data contract (docs/architecture.md §4).

Pure h5py/numpy: runs anywhere, no Genesis needed. Checks structure, shapes,
dtypes, time-axis consistency, and scans for non-finite values. Returns a
Report; the CLI wrapper exits non-zero when issues are found in strict mode.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

REQUIRED_ROOT_ATTRS = ("format_version", "task_id")
REQUIRED_DATA_ATTRS = ("env_args", "problem_info", "total")
REQUIRED_DATASETS = ("actions", "rewards", "dones")
PROPRIO_CHANNELS = ("ee_pos", "ee_ori", "gripper_states", "joint_states")
CAMERA_CHANNELS = ("agentview_rgb", "eye_in_hand_rgb")
ACTION_DIM = 7
# A rendered frame is never legitimately dark: the table fills most of the view
# (measured mean 161-196 of 255, Gate 3). A near-empty frame means the GL path
# silently produced the WSLg D3D12 degenerate image (docs §8) and the whole file
# is worthless — worth failing the run over.
MIN_NONBLACK = 0.05
# obs/link_pose: world position (3) + wxyz quaternion (4) per finger.
LINK_POSE_DIM = 7
QUAT_TOL = 1e-3


@dataclass
class DemoInfo:
    name: str
    num_samples: int = 0
    success: bool | None = None
    channels: dict[str, tuple] = field(default_factory=dict)


@dataclass
class Report:
    path: Path
    format_version: str = ""
    task_id: str = ""
    total: int = 0
    demos: list[DemoInfo] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def summary(self) -> str:
        lines = [
            f"file          : {self.path}",
            f"format        : {self.format_version}  task: {self.task_id}",
            f"demos         : {len(self.demos)} (total attr = {self.total})",
        ]
        if self.demos:
            lens = np.asarray([d.num_samples for d in self.demos])
            n_succ = sum(1 for d in self.demos if d.success)
            lines.append(
                f"episode length: min {lens.min()} / mean {lens.mean():.1f} / max {lens.max()} steps"
                + (f"  | success {n_succ}/{len(self.demos)}" if any(d.success is not None for d in self.demos) else "")
            )
            chans = self.demos[0].channels
            lines.append(f"channels ({len(chans)}):")
            for name, shape in chans.items():
                lines.append(f"  {name:<32} {shape}")
        if self.issues:
            lines.append(f"ISSUES ({len(self.issues)}):")
            lines.extend(f"  - {i}" for i in self.issues)
        else:
            lines.append("OK: no issues found")
        return "\n".join(lines)


def _iter_datasets(group: h5py.Group, prefix: str = ""):
    for key, item in group.items():
        path = f"{prefix}{key}"
        if isinstance(item, h5py.Group):
            yield from _iter_datasets(item, prefix=f"{path}/")
        else:
            yield path, item


def _check_finite(dset: h5py.Dataset, name: str, report: Report, demo: str, *, sample: int = 4096) -> None:
    if dset.dtype.kind != "f":
        return
    arr = dset[()] if dset.size <= sample * 8 else dset[: min(len(dset), 64)]
    arr = np.asarray(arr)
    if not np.isfinite(arr).all():
        report.issues.append(f"{demo}/{name}: contains NaN/Inf")


def _check_camera(
    dset: h5py.Dataset, name: str, report: Report, demo: str, *, resolution: tuple[int, int] | None
) -> None:
    """RGB channel: dtype/shape, plus the two failure modes the renderer has.

    ``render_cameras()`` reusing one read-back buffer would freeze every frame,
    and the WSLg D3D12 path collapses the image into a couple of rows. Both look
    fine structurally, so check pixels: a floor on non-black coverage and a
    frame that actually differs from the first one.
    """
    if dset.dtype != np.uint8:
        report.issues.append(f"{demo}/{name}: dtype {dset.dtype} != uint8")
        return
    if dset.ndim != 4 or dset.shape[-1] != 3:
        report.issues.append(f"{demo}/{name}: shape {dset.shape} != (T,H,W,3)")
        return
    if resolution is not None and tuple(dset.shape[1:3]) != resolution:
        report.issues.append(f"{demo}/{name}: {tuple(dset.shape[1:3])} != configured {resolution}")
    if len(dset) == 0:
        return
    picks = sorted({0, len(dset) // 2, len(dset) - 1})
    frames = {i: np.asarray(dset[i]) for i in picks}
    for i in picks:
        nonblack = float((frames[i].max(axis=-1) > 0).mean())
        if nonblack < MIN_NONBLACK:
            report.issues.append(
                f"{demo}/{name}: frame {i} is {100 * (1 - nonblack):.1f}% black "
                f"(renderer produced no image — see docs §8 on software GL)"
            )
    if len(picks) > 1 and np.array_equal(frames[picks[0]], frames[picks[-1]]):
        report.issues.append(f"{demo}/{name}: first and last frame are identical (frozen buffer?)")


def _layout_grids(f: h5py.File, report: Report) -> dict[str, tuple[int, int]]:
    """``modality -> (rows, cols)`` from ``layouts/modality_grids``; {} if absent.

    Absent means a file written before per-modality grids (format <= 1.0), where one
    ``layouts/probe_grid`` covered every channel. Those files are not shape-checked:
    their row count is not derivable from a per-channel grid, so the check below
    would only invent failures. This mirrors ``tactile_field.modality_grids``, with
    the reporting a validator needs and a reader does not.
    """
    if "layouts" not in f or "modality_grids" not in f["layouts"].attrs:
        return {}
    root = f["layouts"]
    raw = root.attrs["modality_grids"]
    try:
        mapping = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))
    except (TypeError, json.JSONDecodeError) as exc:
        report.issues.append(f"layouts.attrs['modality_grids']: not valid JSON ({exc})")
        return {}
    if not isinstance(mapping, dict):
        report.issues.append(f"layouts.attrs['modality_grids']: not a mapping ({mapping!r})")
        return {}
    grids: dict[str, tuple[int, int]] = {}
    for modality, tag in mapping.items():
        group = root.get(f"grids/{tag}")
        if group is None:
            report.issues.append(f"layouts/grids/{tag}: missing (modality {modality!r} points at it)")
            continue
        if "probe_grid" not in group.attrs:
            report.issues.append(f"layouts/grids/{tag}: missing attr 'probe_grid'")
            continue
        grid = group.attrs["probe_grid"]
        grids[str(modality)] = (int(grid[0]), int(grid[1]))
    return grids


def _channel_formats(f: h5py.File, report: Report) -> dict[str, dict]:
    """``modality -> {features, frame, shape}`` from ``layouts.channel_formats``.

    Empty for legacy files (format <= 1.5): those keep the old grid-derived
    shape check. New files declare the derived channels explicitly, which is the
    only way to tell a per-finger 6D force from a per-probe vector channel.
    """
    if "layouts" not in f or "channel_formats" not in f["layouts"].attrs:
        return {}
    raw = f["layouts"].attrs["channel_formats"]
    try:
        mapping = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))
    except (TypeError, json.JSONDecodeError) as exc:
        report.issues.append(f"layouts.attrs['channel_formats']: not valid JSON ({exc})")
        return {}
    if not isinstance(mapping, dict):
        report.issues.append(f"layouts.attrs['channel_formats']: not a mapping ({mapping!r})")
        return {}
    out: dict[str, dict] = {}
    for modality, spec in mapping.items():
        if not isinstance(spec, dict):
            report.issues.append(f"layouts.attrs['channel_formats'][{modality!r}]: not a mapping")
            continue
        try:
            features = int(spec["features"])
            frame = str(spec["frame"])
            shape = tuple(int(v) for v in spec["shape"])
        except (KeyError, TypeError, ValueError) as exc:
            report.issues.append(f"layouts.attrs['channel_formats'][{modality!r}]: unreadable ({exc})")
            continue
        if features < 1 or not shape or any(v < 0 for v in shape):
            report.issues.append(
                f"layouts.attrs['channel_formats'][{modality!r}]: bad features/shape "
                f"({features}, {shape})"
            )
            continue
        if frame not in ("link", "pad_tangent"):
            report.issues.append(
                f"layouts.attrs['channel_formats'][{modality!r}]: unknown frame {frame!r}"
            )
        out[str(modality)] = {"features": features, "frame": frame, "shape": shape}
    return out


def _check_grid_shapes(
    obs: h5py.Group,
    demo_name: str,
    grids: dict[str, tuple[int, int]],
    formats: dict[str, dict],
    n_links: int,
    t_ref: int | None,
    report: Report,
) -> None:
    """Every tactile channel's shape must match its declared format.

    Format 1.6 declares ``force (n_fingers, 6)`` and ``elastomer
    (n_fingers*P, 2)``. Legacy files (no ``channel_formats``) keep the old
    ``n_links * prod(grid)`` rule. Either way this catches a channel silently
    regrouped onto the wrong shape, which nothing downstream can detect from
    the values alone.
    """
    for modality, (rows, cols) in grids.items():
        ch = f"tactile/{modality}"
        if ch not in obs:
            continue
        dset = obs[ch]
        shape = tuple(dset.shape)
        n_p = rows * cols
        spec = formats.get(modality)
        if spec is not None:
            features = int(spec["features"])
            if modality == "force":
                expect = (shape[0], n_links, features)
            elif features == 1:
                expect = (shape[0], n_links * n_p)
            else:
                expect = (shape[0], n_links * n_p, features)
            if shape != expect:
                report.issues.append(
                    f"{demo_name}/obs/{ch}: shape {shape} != {expect} "
                    f"(declared format: {spec['features']} features, {spec['frame']} frame)"
                )
            declared = tuple(spec["shape"])
            if declared != expect[1:]:
                report.issues.append(
                    f"{demo_name}/obs/{ch}: channel_formats shape {declared} != "
                    f"expected {expect[1:]} ({n_links} links x {rows}x{cols} grid)"
                )
        else:
            if dset.ndim == 2:
                expect = (shape[0], n_links * n_p)
            elif dset.ndim == 3 and shape[2] == 3:
                expect = (shape[0], n_links * n_p, 3)
            else:
                report.issues.append(f"{demo_name}/obs/{ch}: shape {shape} is not (T, n_links*P[, 3])")
                continue
            if shape != expect:
                report.issues.append(
                    f"{demo_name}/obs/{ch}: shape {shape} != {expect} "
                    f"({n_links} links x {rows}x{cols} grid)"
                )
        if t_ref is not None and shape[0] != t_ref:
            report.issues.append(f"{demo_name}/obs/{ch}: T={shape[0]} != {t_ref}")
        if dset.dtype != np.float32:
            report.issues.append(f"{demo_name}/obs/{ch}: dtype {dset.dtype} != float32")


def _check_link_pose(obs: h5py.Group, demo_name: str, n_links: int, t_ref: int | None, report: Report) -> None:
    """``obs/link_pose`` is ``(T, n_links, 7)`` float32: pos_world + quat_wxyz."""
    if "link_pose" not in obs:
        return
    dset = obs["link_pose"]
    shape = tuple(dset.shape)
    if shape != (shape[0], n_links, LINK_POSE_DIM):
        report.issues.append(
            f"{demo_name}/obs/link_pose: shape {shape} != (T, {n_links}, {LINK_POSE_DIM})"
        )
        return
    if t_ref is not None and shape[0] != t_ref:
        report.issues.append(f"{demo_name}/obs/link_pose: T={shape[0]} != {t_ref}")
    if dset.dtype != np.float32:
        report.issues.append(f"{demo_name}/obs/link_pose: dtype {dset.dtype} != float32")
    quat = dset[..., 3:].astype(np.float64)
    norms = np.linalg.norm(quat, axis=-1)
    if norms.size and float(np.abs(norms - 1.0).max()) > QUAT_TOL:
        report.issues.append(
            f"{demo_name}/obs/link_pose: quaternion norm off by {float(np.abs(norms - 1.0).max()):.2e}"
        )


def _check_height_map(
    f: h5py.File,
    obs: h5py.Group,
    demo_name: str,
    n_links: int,
    t_ref: int | None,
    report: Report,
) -> None:
    """Format-1.3 ``obs/tactile/height_map`` — checked only when it exists.

    Beyond shape/dtype/finiteness, the map is checked against a *re-synthesis
    from the file's own markers* — the reader path in ``tools/height_map.py``.
    That is the only exact test available: the map is the marker field resampled
    onto a raster, and since the raster does not land on every marker (9x7 ->
    64x64 loses ~1.5% of a lone spike's peak), comparing extrema would be
    data-dependent. Any mangled axis order, frame, sign, grid or size makes the
    two arrays disagree by far more than the interpolation's own noise.
    """
    key = "tactile/height_map"
    if obs is None or key not in obs:
        return
    from sim.tools.height_map import height_map_meta, synth_from_markers  # local: keep this module import-light

    meta = height_map_meta(f)
    if meta is None:
        report.issues.append(f"{demo_name}/obs/{key}: present but layouts/ has no 'height_map' metadata")
        return
    dset = obs[key]
    shape = tuple(dset.shape)
    size = tuple(int(v) for v in meta.get("size", (0, 0)))
    if dset.ndim != 4 or shape[1] != n_links or tuple(shape[2:]) != size:
        report.issues.append(f"{demo_name}/obs/{key}: shape {shape} != (T, {n_links}, {size[0]}, {size[1]})")
        return
    if t_ref is not None and shape[0] != t_ref:
        report.issues.append(f"{demo_name}/obs/{key}: T={shape[0]} != {t_ref}")
    if dset.dtype != np.float32:
        report.issues.append(f"{demo_name}/obs/{key}: dtype {dset.dtype} != float32")
    if "tactile/elastomer" not in obs:
        report.issues.append(f"{demo_name}/obs/{key}: present but obs/tactile/elastomer (its source) is not")
        return
    stored = dset[:]
    if not np.all(np.isfinite(stored)):
        report.issues.append(f"{demo_name}/obs/{key}: contains non-finite values")
        return
    try:
        ref = synth_from_markers(f, demo_name, size=size, interpolate=meta.get("interpolation", "cubic"))
    except (KeyError, ValueError) as exc:
        report.issues.append(f"{demo_name}/obs/{key}: cannot re-synthesize from the markers ({exc})")
        return
    if ref.shape != stored.shape:
        report.issues.append(f"{demo_name}/obs/{key}: re-synthesis gives {ref.shape}, stored {stored.shape}")
    elif not np.allclose(stored, ref, rtol=1e-5, atol=1e-12):
        report.issues.append(
            f"{demo_name}/obs/{key}: differs from the map re-synthesized from obs/tactile/elastomer "
            f"(max |delta| {float(np.abs(stored - ref).max()):.3e} m, field peak "
            f"{float(np.abs(ref).max()):.3e} m) — frame, axis, sign or grid mismatch"
        )


def _check_multi_object(
    obs: h5py.Group | None,
    demo: h5py.Group,
    demo_name: str,
    t_ref: int | None,
    n_declared: int | None,
    object_grid: tuple[int, ...] | None,
    report: Report,
) -> None:
    """Format-1.2 object channels — checked only when ``obs/object_pose`` exists.

    Files written before multi-object scenes have neither key, so they skip these
    checks entirely rather than fail them (same policy as ``modality_grids``
    above): absence is history, not an issue.
    """
    if obs is None or "object_pose" not in obs:
        return
    dset = obs["object_pose"]
    shape = tuple(dset.shape)
    if dset.ndim != 3 or shape[-1] != LINK_POSE_DIM:
        report.issues.append(f"{demo_name}/obs/object_pose: shape {shape} != (T, N, {LINK_POSE_DIM})")
        return
    n = int(shape[1])
    if t_ref is not None and shape[0] != t_ref:
        report.issues.append(f"{demo_name}/obs/object_pose: T={shape[0]} != {t_ref}")
    if dset.dtype != np.float32:
        report.issues.append(f"{demo_name}/obs/object_pose: dtype {dset.dtype} != float32")
    quat = dset[..., 3:].astype(np.float64)
    norms = np.linalg.norm(quat, axis=-1)
    if norms.size and float(np.abs(norms - 1.0).max()) > QUAT_TOL:
        report.issues.append(
            f"{demo_name}/obs/object_pose: quaternion norm off by {float(np.abs(norms - 1.0).max()):.2e}"
        )
    if n_declared is not None and n != n_declared:
        report.issues.append(
            f"{demo_name}/obs/object_pose: {n} objects != env_args.objects ({n_declared} entries)"
        )
    if "target_index" not in demo.attrs:
        report.issues.append(f"{demo_name}: missing attr 'target_index' (obs/object_pose present)")
    else:
        t = int(demo.attrs["target_index"])
        if not 0 <= t < n:
            report.issues.append(f"{demo_name}: target_index {t} outside [0, {n})")
    if "temperature/objects_grid" in obs and object_grid is not None:
        g = obs["temperature/objects_grid"]
        gshape = tuple(g.shape)
        expect = (gshape[0] if gshape else 0, n, *object_grid)
        if gshape != expect:
            report.issues.append(
                f"{demo_name}/obs/temperature/objects_grid: shape {gshape} != {expect}"
            )
        if t_ref is not None and gshape and gshape[0] != t_ref:
            report.issues.append(f"{demo_name}/obs/temperature/objects_grid: T={gshape[0]} != {t_ref}")


def _check_finger_temperature(
    f: h5py.File,
    obs: h5py.Group | None,
    demo_name: str,
    n_links: int,
    t_ref: int | None,
    report: Report,
) -> None:
    """``obs/temperature/fingertip_grid`` — the format-1.5 lumped measured cell.

    Files written by the official-route configs (docs §9-R36) declare the sensor
    in ``layouts.attrs['temperature_sensor']``: the engine's default
    ``grid_size=(1,1,1)`` per finger, ``record: measured`` with the finger
    sensors' RC/noise/quantization. Older files (formats <= 1.4) have no such
    attr and skip silently (same absence-is-history policy as
    ``modality_grids``). When the declaration says the readings are quantized
    (``record: measured`` with ``resolution > 0``) every recorded value must sit
    on the resolution lattice — that is the check which catches a file that
    silently fell back to ground-truth recording.
    """
    key = "temperature/fingertip_grid"
    if obs is None or key not in obs or "layouts" not in f:
        return
    root = f["layouts"]
    raw = root.attrs.get("temperature_sensor")
    if raw is None:
        return
    try:
        meta = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))
        grid = tuple(int(v) for v in meta["grid"])
        record = str(meta.get("record", ""))
        res = float(meta.get("resolution", 0.0))
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        report.issues.append(f"layouts.attrs['temperature_sensor']: unreadable ({exc})")
        return
    if len(grid) != 3 or any(n < 1 for n in grid):
        report.issues.append(f"layouts.attrs['temperature_sensor']: bad grid {grid}")
        return
    # The declaration and the legacy-shape triple are two record sites of the
    # same numbers; they must not drift apart.
    tg = root.attrs.get("temperature_grids")
    if tg is not None:
        finger = tuple(int(v) for v in np.asarray(tg).ravel()[:3])
        if finger != grid:
            report.issues.append(
                f"layouts.attrs: temperature_grids finger triple {finger} != temperature_sensor grid {grid}"
            )

    dset = obs[key]
    shape = tuple(dset.shape)
    if n_links >= 1 and (dset.ndim != 5 or shape[1] != n_links or tuple(shape[2:]) != grid):
        report.issues.append(
            f"{demo_name}/obs/{key}: shape {shape} != (T, {n_links}, {grid[0]}, {grid[1]}, {grid[2]})"
        )
    if t_ref is not None and shape[0] != t_ref:
        report.issues.append(f"{demo_name}/obs/{key}: T={shape[0]} != {t_ref}")
    if dset.dtype != np.float32:
        report.issues.append(f"{demo_name}/obs/{key}: dtype {dset.dtype} != float32")

    if record == "measured" and res > 0.0 and dset.size:
        vals = dset[:].astype(np.float64).ravel()
        steps = vals / res
        # float32 storage of a lattice value leaves ~1e-5 of a step; NaNs are
        # handled by _check_finite and compare False here.
        worst = float(np.abs(steps - np.round(steps)).max())
        if worst > 1e-3:
            report.issues.append(
                f"{demo_name}/obs/{key}: values are not on the {res:g} C resolution lattice "
                f"(worst {worst:.2e} of a step) though the sensor declares record=measured — "
                "was this file written with record_ground_truth: true?"
            )


def validate_file(path: str | Path, *, strict_channels: bool = True, sample: int = 4096) -> Report:
    """Validate one ThermTacVLA HDF5 file.

    Parameters
    ----------
    strict_channels : require the LIBERO-canonical proprio channels and the
        presence of at least one tactile or temperature channel per demo.
    sample : cap on the number of demos whose data is numerically scanned.
    """
    path = Path(path)
    report = Report(path=path)

    with h5py.File(path, "r") as f:
        for attr in REQUIRED_ROOT_ATTRS:
            if attr not in f.attrs:
                report.issues.append(f"root: missing attr {attr!r}")
        report.format_version = str(f.attrs.get("format_version", ""))
        report.task_id = str(f.attrs.get("task_id", ""))

        if "data" not in f:
            report.issues.append("missing 'data' group")
            return report
        data = f["data"]
        for attr in REQUIRED_DATA_ATTRS:
            if attr not in data.attrs:
                report.issues.append(f"data: missing attr {attr!r}")
        env_args: dict = {}
        for attr in ("env_args", "problem_info"):
            if attr in data.attrs:
                try:
                    parsed = json.loads(data.attrs[attr])
                    if attr == "env_args" and isinstance(parsed, dict):
                        env_args = parsed
                except (TypeError, json.JSONDecodeError) as exc:
                    report.issues.append(f"data.attrs[{attr!r}]: not valid JSON ({exc})")
        report.total = int(data.attrs.get("total", -1))

        # The camera contract is declared in env_args, so a file says for itself
        # whether RGB is expected — v1 files (cameras disabled) are unaffected.
        cam_cfg = env_args.get("cameras") or {}
        cameras_enabled = bool(cam_cfg.get("enabled"))
        cam_res = cam_cfg.get("resolution")
        cam_res = tuple(int(v) for v in cam_res) if isinstance(cam_res, (list, tuple)) and len(cam_res) == 2 else None
        declared = tuple(cam_cfg.get("channels") or ())

        # Per-modality probe grids (format 1.1): the authority on every tactile
        # channel's row count, and what makes the shape checks below possible.
        # Files without them (format <= 1.0) skip those checks entirely.
        grids = _layout_grids(f, report)
        formats = _channel_formats(f, report)
        n_links = int(f["layouts"].attrs.get("n_links", -1)) if "layouts" in f else -1
        if grids and n_links < 1:
            report.issues.append("layouts: missing or invalid attr 'n_links' (per-modality grids present)")

        # Multi-object declarations (format 1.2); None on older files, which then
        # skip the matching checks below.
        obj_decl = env_args.get("objects")
        n_objects = len(obj_decl) if isinstance(obj_decl, list) else None
        obj_grid = (env_args.get("sensors") or {}).get("temperature", {}).get("object_grid")
        obj_grid = tuple(int(v) for v in obj_grid) if isinstance(obj_grid, (list, tuple)) and obj_grid else None

        demo_names = sorted((k for k in data if k.startswith("demo_")), key=lambda s: int(s.split("_")[1]))
        if report.total != len(demo_names):
            report.issues.append(f"total attr ({report.total}) != demo groups ({len(demo_names)})")
        if not demo_names:
            report.issues.append("no demo_N groups found")

        for i, demo_name in enumerate(demo_names):
            demo = data[demo_name]
            info = DemoInfo(name=demo_name)
            info.num_samples = int(demo.attrs.get("num_samples", -1))
            if "success" in demo.attrs:
                info.success = bool(demo.attrs["success"])

            t_ref: int | None = None
            for name in REQUIRED_DATASETS:
                if name not in demo:
                    report.issues.append(f"{demo_name}: missing dataset {name!r}")
            for name, dset in _iter_datasets(demo, prefix=""):
                if name.startswith("events/"):
                    continue
                shape = tuple(dset.shape)
                info.channels[name] = shape
                if name in REQUIRED_DATASETS and len(shape) >= 1:
                    t_ref = t_ref if t_ref is not None else shape[0]
                    if shape[0] != t_ref:
                        report.issues.append(f"{demo_name}/{name}: T={shape[0]} != {t_ref}")
            if info.num_samples > 0 and t_ref is not None and info.num_samples != t_ref:
                report.issues.append(f"{demo_name}: num_samples ({info.num_samples}) != T ({t_ref})")

            if "actions" in demo:
                act = demo["actions"]
                if act.shape[1:] != (ACTION_DIM,):
                    report.issues.append(f"{demo_name}/actions: shape {act.shape} != (T,{ACTION_DIM})")
                if act.dtype != np.float32:
                    report.issues.append(f"{demo_name}/actions: dtype {act.dtype} != float32")
            if "dones" in demo:
                dones = demo["dones"]
                if dones.dtype != np.bool_:
                    report.issues.append(f"{demo_name}/dones: dtype {dones.dtype} != bool")
                if len(dones) and not bool(dones[-1]):
                    report.issues.append(f"{demo_name}/dones: last step is not done")

            obs = demo.get("obs")
            if obs is not None:
                obs_names = [n for n, _ in _iter_datasets(obs)]
                if strict_channels:
                    for ch in PROPRIO_CHANNELS:
                        if ch not in obs_names:
                            report.issues.append(f"{demo_name}/obs: missing channel {ch!r}")
                    if not any(n.startswith(("tactile/", "temperature/")) for n in obs_names):
                        report.issues.append(f"{demo_name}/obs: no tactile/ or temperature/ channels")
                if cameras_enabled:
                    for ch in CAMERA_CHANNELS:
                        if ch not in obs_names:
                            report.issues.append(f"{demo_name}/obs: cameras.enabled but channel {ch!r} missing")
                for ch in CAMERA_CHANNELS:
                    if ch in obs_names:
                        _check_camera(obs[ch], ch, report, demo_name, resolution=cam_res)
                if declared:
                    for ch in declared:
                        if ch not in obs_names:
                            report.issues.append(f"{demo_name}/obs: env_args declares channel {ch!r} but it is absent")
                if grids and n_links >= 1:
                    _check_grid_shapes(obs, demo_name, grids, formats, n_links, t_ref, report)
                    _check_link_pose(obs, demo_name, n_links, t_ref, report)
                    _check_height_map(f, obs, demo_name, n_links, t_ref, report)
                _check_multi_object(obs, demo, demo_name, t_ref, n_objects, obj_grid, report)
                _check_finger_temperature(f, obs, demo_name, n_links, t_ref, report)
            elif strict_channels:
                report.issues.append(f"{demo_name}: missing obs group")

            if i < sample:
                for name, dset in _iter_datasets(demo):
                    if dset.dtype.kind == "f":
                        _check_finite(dset, name, report, demo_name, sample=sample)

            report.demos.append(info)

    return report

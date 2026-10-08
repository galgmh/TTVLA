"""Synthesize the dense tactile height map from the recorded marker field.

Pure h5py + numpy (+ scipy for the spline), same tier as ``validate.py`` and
``tactile_field.py``: a training dataloader can use this without importing
genesis.

Why synthesis and not rendering: this project's gel is a *virtual* sensor
(``ElastomerTaxel`` = SDF queries + mesh sampling), so the scene holds no
deformable geometry for a camera to photograph — there is literally nothing to
render. UniVTAC can render ``press_depth`` because it simulates a FEM gelpad
with an in-finger camera; the equivalent physical quantity here — the
out-of-plane displacement of the pad surface — is obtained by interpolating the
marker samples onto a dense grid (UniVTAC's markers are likewise attached to its
gel surface by interpolation). See docs §9-R25.

The map is a *pure resampling* of ``obs/tactile/elastomer``: an interpolating
spline through the marker samples, evaluated on a regular raster. It adds no
information — a 9x7 field cannot carry detail its 63 samples never saw — and it
is bounded by the samples it came from (the spline overshoot is clamped away).
The raster does not land on every marker, so the map reproduces the field's
extent and range rather than every sample value; validate.py checks it the only
way that is exact regardless of field shape — by re-synthesizing from the stored
markers and comparing the arrays.
"""

from __future__ import annotations

import json
from typing import Any

import h5py
import numpy as np

from tempvla.rotations import quat_to_matrix

MARKER_MODALITY = "elastomer"
DEFAULT_SIZE = (64, 64)


# ------------------------------------------------------------------- synthesis


def upsample_field(plane: np.ndarray, size: tuple[int, int], interpolate: str = "cubic") -> np.ndarray:
    """``(ny, nx)`` marker plane -> ``(H, W)`` float32 image, cell-centred.

    Pixel ``(r, c)`` samples the field at marker index ``(r + 0.5)·ny/H - 0.5``
    (columns likewise): the image spans the grid's *cells*, i.e. half a cell of
    margin beyond the outermost markers. That is the extent model the renderer
    already uses for these fields (``gel_render.grid_m_per_px`` /
    ``lattice_nodes``), so a height-map panel and a coarse-field panel of the
    same grid cover the same patch of pad and the marker overlay lines up on
    both. The margin is a flat extension of the outer marker line: the sample
    coordinates are clamped to the marker range, so a margin pixel evaluates
    *exactly on* the outermost row/column. (Clamping the coordinate, rather than
    only passing ``mode="nearest"``, matters: scipy prefilters the spline with a
    mirror boundary, so at fractional coordinates just outside the range the
    interpolation would otherwise ring by a few percent of the field.)

    Rows follow grid axis 0, columns axis 1. The spline passes through every
    marker sample but may overshoot between them, and the raster need not land on
    a sample at all (a lone spike loses ~1.5% of its peak at 9x7 -> 64x64), so the
    result is clamped to the sample min/max: the map never claims a displacement
    the sensor did not read.
    """
    plane = np.asarray(plane, dtype=np.float64)
    ny, nx = plane.shape
    rows, cols = int(size[0]), int(size[1])
    ys = np.clip((np.arange(rows) + 0.5) * ny / rows - 0.5, 0.0, ny - 1.0)
    xs = np.clip((np.arange(cols) + 0.5) * nx / cols - 0.5, 0.0, nx - 1.0)
    coords = np.stack(np.meshgrid(ys, xs, indexing="ij"))  # (2, H, W)
    try:
        from scipy.ndimage import map_coordinates  # lazy: keep module import light
    except ImportError:  # pragma: no cover - scipy ships with genesis's stack
        out = _bilinear(plane, ys, xs)
    else:
        out = map_coordinates(plane, coords, order=3 if interpolate == "cubic" else 1, mode="nearest")
    return np.clip(out, plane.min(), plane.max()).astype(np.float32)


def _bilinear(plane: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """numpy-only fallback (no scipy): bilinear at the same sample coordinates
    (``np.interp`` clamps outside the range, which is the flat margin)."""
    ny, nx = plane.shape
    row = np.stack([np.interp(xs, np.arange(nx), plane[i]) for i in range(ny)], axis=0)  # (ny, W)
    return np.stack([np.interp(ys, np.arange(ny), row[:, j]) for j in range(row.shape[1])], axis=1)


def synthesize(planes: np.ndarray, size: tuple[int, int], interpolate: str = "cubic") -> np.ndarray:
    """``(..., ny, nx)`` height planes -> ``(..., H, W)`` float32 maps."""
    planes = np.asarray(planes, dtype=np.float64)
    flat = planes.reshape(-1, *planes.shape[-2:])
    out = np.stack([upsample_field(p, size, interpolate) for p in flat])
    return out.reshape(*planes.shape[:-2], *out.shape[-2:])


def pad_normal_planes(
    disp: np.ndarray, quats_wxyz: np.ndarray | None, normals: Any, grid: tuple[int, int]
) -> np.ndarray:
    """``(F, P, 3)`` displacement -> ``(F, ny, nx)`` pad-normal height planes.

    ``disp`` is the recorded channel: world frame when ``world_frame: true``
    (then ``quats_wxyz`` ``(F, 4)`` is required and the vectors are rotated
    back), link frame otherwise (``quats_wxyz=None``). Exact mirror of
    ``tactile_field.pad_normal_field`` — same projection onto the stored pad
    normal, no extra sign flip — so a map synthesized here and one synthesized
    by a reader from the stored markers agree exactly (validate.py checks that).
    """
    disp = np.asarray(disp, dtype=np.float64)
    n_fingers, n_p, _ = disp.shape
    ny, nx = int(grid[0]), int(grid[1])
    if ny * nx != n_p:
        raise ValueError(f"displacement has {n_p} probes/finger, grid {ny}x{nx} implies {ny * nx}")
    if quats_wxyz is not None:
        rot = quat_to_matrix(np.asarray(quats_wxyz, dtype=np.float64))  # (F, 3, 3)
        disp = np.einsum("fji,fpj->fpi", rot, disp)  # v_link = R^T @ v_world
    nrm = np.asarray(normals, dtype=np.float64).reshape(n_fingers, ny, nx, 3)
    return (disp.reshape(n_fingers, ny, nx, 3) * nrm).sum(axis=-1)


# ---------------------------------------------------------------------- reader


def height_map_meta(f: h5py.File) -> dict[str, Any] | None:
    """The ``layouts/`` attr written by the generator, or None for old files."""
    if "layouts" not in f or "height_map" not in f["layouts"].attrs:
        return None
    raw = f["layouts"].attrs["height_map"]
    try:
        return json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))
    except (TypeError, json.JSONDecodeError):
        return None


def read_height_map(f: h5py.File, demo: str) -> np.ndarray | None:
    """``(T, n_fingers, H, W)`` float32 metres, or None when not recorded."""
    key = f"data/{demo}/obs/tactile/height_map"
    if key not in f:
        return None
    return f[key][:].astype(np.float32)


def synth_from_markers(
    f: h5py.File, demo: str, *, size: tuple[int, int] = DEFAULT_SIZE, interpolate: str = "cubic"
) -> np.ndarray:
    """Synthesize the height map from a file's marker field.

    Works on any file that carries ``obs/tactile/elastomer`` + ``layouts/`` —
    including files written before the channel existed.
    """
    from tempvla.tools import tactile_field as tf

    grid = tf.modality_grid(f, MARKER_MODALITY)
    if grid is None:
        raise ValueError("layouts/ has no probe grid for the marker modality")
    disp = tf.read_markers(f, demo, MARKER_MODALITY, frame="link")  # (T, F, ny, nx, 3)
    _, normals = tf.finger_geometry(f, MARKER_MODALITY)
    if normals is None:
        raise ValueError("layouts/probe_local_normal missing — cannot project onto the pad normal")
    nrm = np.stack([n.reshape(grid[0], grid[1], 3) for n in normals])
    planes = (disp * nrm[None]).sum(axis=-1)
    return synthesize(planes, size, interpolate)


def height_map_or_synth(f: h5py.File, demo: str) -> np.ndarray:
    """The recorded height map when present, else synthesized from the markers.

    The synthesis parameters fall back to the ``layouts/`` metadata, then to the
    module defaults, so a legacy file (no metadata) still gets a sensible map.
    """
    stored = read_height_map(f, demo)
    if stored is not None:
        return stored
    meta = height_map_meta(f) or {}
    size = tuple(meta.get("size", DEFAULT_SIZE))
    return synth_from_markers(f, demo, size=size, interpolate=meta.get("interpolation", "cubic"))

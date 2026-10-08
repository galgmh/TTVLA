"""Reader for the recorded marker field (``obs/tactile/elastomer``).

Pure h5py + numpy, same tier as ``validate.py``: a training dataloader can use
this without importing genesis or building a scene.

Storage contract (docs §5.2): the channel holds per-marker *displacement*
``(T, n_fingers * prod(grid), 3)`` float32, the undeformed marker positions live
in ``layouts/`` (root alias = the primary modality's grid, ``layouts/grids/<tag>/``
per grid), and ``obs/link_pose`` holds the finger pose at each step. Together
that is everything a marker-field consumer needs:

    displacement  ->  positions      reference + displacement
    world frame   ->  link frame     rotate by obs/link_pose

and the link frame is where the marker field *is* a GelSight-like height field
(project onto ``layouts/probe_local_normal`` — out-of-plane is not an axis of the
link frame, it is -y on the Panda fingers).

Tabero (arXiv:2605.27886) consumes a marker field as ``(H+1, N, 2)`` of 2D
positions — the reference frame followed by H history frames. ``as_tabero_sequence``
emits the same shape in 3D: ``(T+1, n_fingers*P, 3)``.
"""

from __future__ import annotations

import json
from typing import Any

import h5py
import numpy as np

from tempvla.rotations import quat_to_matrix

MARKER_MODALITY = "elastomer"


# ------------------------------------------------------------------ metadata


def modality_grids(f: h5py.File) -> dict[str, tuple[int, int]]:
    """``modality -> (ny, nx)`` from ``layouts/modality_grids``.

    Empty for files written before per-modality grids existed; those have one
    ``layouts/probe_grid`` shared by every channel (use :func:`modality_grid`).
    """
    if "layouts" not in f or "modality_grids" not in f["layouts"].attrs:
        return {}
    raw = f["layouts"].attrs["modality_grids"]
    mapping = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))
    grids: dict[str, tuple[int, int]] = {}
    for modality, tag in mapping.items():
        group = f.get(f"layouts/grids/{tag}")
        if group is None:
            continue
        grid = group.attrs["probe_grid"]
        grids[str(modality)] = (int(grid[0]), int(grid[1]))
    return grids


def layout_group(f: h5py.File, modality: str) -> h5py.Group | None:
    """The group holding ``modality``'s probe geometry, or None if absent.

    New files: ``layouts/grids/<tag>/``. Legacy single-grid files: the root of
    ``layouts/``.
    """
    if "layouts" not in f:
        return None
    root = f["layouts"]
    if "modality_grids" in root.attrs:
        raw = root.attrs["modality_grids"]
        mapping = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))
        tag = mapping.get(modality)
        if tag is None or "grids" not in root or tag not in root["grids"]:
            return None
        return root["grids"][tag]
    return root


def modality_grid(f: h5py.File, modality: str) -> tuple[int, int] | None:
    """``(ny, nx)`` for one channel, or None when the file stores no grid."""
    group = layout_group(f, modality)
    if group is None or "probe_grid" not in group.attrs:
        return None
    grid = group.attrs["probe_grid"]
    return (int(grid[0]), int(grid[1]))


def world_frame(f: h5py.File) -> bool:
    """Whether the vector channels were rotated into the world frame.

    Defaults to True: every file written before ``world_frame`` was recorded as
    an attribute used the world frame.
    """
    if "layouts" in f and "world_frame" in f["layouts"].attrs:
        return bool(f["layouts"].attrs["world_frame"])
    return True


def finger_geometry(f: h5py.File, modality: str) -> tuple[list[np.ndarray], list[np.ndarray] | None]:
    """Per-finger ``(probe_local_pos (P,3), probe_local_normal (P,3))`` in link order."""
    group = layout_group(f, modality)
    if group is None or "probe_local_pos" not in group:
        return [], None
    pos = group["probe_local_pos"][:].astype(np.float64)
    nrm = group["probe_local_normal"][:].astype(np.float64) if "probe_local_normal" in group else None
    link = (
        group["probe_link_index"][:]
        if "probe_link_index" in group
        else np.zeros(pos.shape[0], dtype=np.int32)
    )
    n_fingers = int(link.max()) + 1 if link.size else 0
    positions = [pos[link == k] for k in range(n_fingers)]
    normals = [nrm[link == k] for k in range(n_fingers)] if nrm is not None else None
    return positions, normals


def link_names(f: h5py.File) -> list[str]:
    """Finger link names, in the layout order the channels are concatenated in."""
    if "layouts/link_names" not in f:
        return []
    return [n.decode() if isinstance(n, bytes) else str(n) for n in f["layouts/link_names"][:]]


# ---------------------------------------------------------------------- read


def read_link_pose(f: h5py.File, demo: str) -> np.ndarray | None:
    """``(T, n_fingers, 7)`` world position + wxyz quaternion, or None if absent."""
    key = f"data/{demo}/obs/link_pose"
    if key not in f:
        return None
    return f[key][:].astype(np.float64)


def read_markers(
    f: h5py.File,
    demo: str,
    modality: str = MARKER_MODALITY,
    *,
    frame: str = "link",
) -> np.ndarray:
    """Per-marker displacement as ``(T, n_fingers, ny, nx, 3)`` float64, metres.

    ``frame="link"`` (the default, and what the height field needs) rotates a
    world-frame recording back with ``obs/link_pose``. ``frame="world"`` returns
    whatever the file recorded.
    """
    key = f"data/{demo}/obs/tactile/{modality}"
    if key not in f:
        raise KeyError(f"{demo} has no obs/tactile/{modality}")
    arr = f[key][:].astype(np.float64)
    grid = modality_grid(f, modality)
    if grid is None:
        raise ValueError(
            f"{key}: no probe grid in layouts/ (need `modality_grids` or a root `probe_grid`)"
        )
    n_p = grid[0] * grid[1]
    if arr.ndim != 3 or arr.shape[1] % n_p:
        raise ValueError(f"{key}: {arr.shape} is not (T, n_fingers*{n_p}, 3)")
    n_fingers = arr.shape[1] // n_p
    out = arr.reshape(arr.shape[0], n_fingers, grid[0], grid[1], 3)
    if frame == "world" or not world_frame(f):
        return out
    pose = read_link_pose(f, demo)
    if pose is None:
        raise ValueError(
            f"{key}: recorded in the world frame with no obs/link_pose — the "
            "link-frame field is not recoverable (regenerate, docs §9-R8)"
        )
    if pose.shape[:2] != out.shape[:2]:
        raise ValueError(f"{key}: obs/link_pose {pose.shape[:2]} does not match the marker field {out.shape[:2]}")
    # v_world = R @ v_link  =>  v_link = Rᵀ @ v_world
    rot = quat_to_matrix(pose[:, :, 3:7])          # (T, n_fingers, 3, 3)
    flat = out.reshape(out.shape[0], n_fingers, -1, 3)
    return np.einsum("tfji,tfpj->tfpi", rot, flat).reshape(out.shape)


def reference_positions(f: h5py.File, modality: str, n_fingers: int, grid: tuple[int, int]) -> np.ndarray:
    """Undeformed marker positions ``(n_fingers, ny, nx, 3)``, link frame, metres."""
    positions, _ = finger_geometry(f, modality)
    if len(positions) != n_fingers:
        raise ValueError(f"layouts/ holds {len(positions)} fingers, the channel has {n_fingers}")
    return np.stack([p.reshape(grid[0], grid[1], 3) for p in positions])


def marker_positions(
    f: h5py.File, demo: str, modality: str = MARKER_MODALITY, *, frame: str = "link"
) -> np.ndarray:
    """Marker positions ``(T, n_fingers, ny, nx, 3)``: reference + displacement."""
    disp = read_markers(f, demo, modality, frame=frame)
    grid = modality_grid(f, modality)
    ref = reference_positions(f, modality, disp.shape[1], grid)
    return ref[None] + disp


def pad_normal_field(
    f: h5py.File, demo: str, modality: str = MARKER_MODALITY
) -> np.ndarray:
    """Out-of-plane displacement ``(T, n_fingers, ny, nx)``, metres — the height field.

    This is the projection onto the stored pad normal (``probe_local_normal``),
    so it is only defined in the link frame; a world-frame vector has been
    rotated away from it.
    """
    disp = read_markers(f, demo, modality, frame="link")
    _, normals = finger_geometry(f, modality)
    if normals is None:
        raise ValueError("layouts/probe_local_normal missing — cannot project onto the pad normal")
    grid = modality_grid(f, modality)
    nrm = np.stack([n.reshape(grid[0], grid[1], 3) for n in normals])
    return (disp * nrm[None]).sum(axis=-1)


def as_tabero_sequence(
    f: h5py.File, demo: str, modality: str = MARKER_MODALITY, *, frame: str = "link"
) -> np.ndarray:
    """``(T+1, n_fingers*P, 3)``: the reference pose followed by every step's positions.

    Tabero's marker field is ``M in R^{(H+1) x N x 2}`` — a reference frame plus
    H history frames. Ours is 3D link-frame positions over the whole episode;
    slice a ``(H+1)`` window along axis 0 for training.
    """
    disp = read_markers(f, demo, modality, frame=frame)
    grid = modality_grid(f, modality)
    ref = reference_positions(f, modality, disp.shape[1], grid)
    seq = np.concatenate([ref[None], ref[None] + disp], axis=0)   # (T+1, F, ny, nx, 3)
    return seq.reshape(seq.shape[0], -1, 3)


def summary(f: h5py.File) -> dict[str, Any]:
    """Channel inventory for one file: grids, rows, frames, marker counts."""
    grids = modality_grids(f)
    out: dict[str, Any] = {
        "format_version": f.attrs.get("format_version"),
        "world_frame": world_frame(f),
        "link_names": link_names(f),
        "per_modality_grid": {m: list(g) for m, g in grids.items()},
    }
    for modality, grid in grids.items():
        n_rows = None
        for demo in sorted(f["data"], key=lambda k: int(k.split("_")[1])):
            key = f"data/{demo}/obs/tactile/{modality}"
            if key in f:
                n_rows = int(f[key].shape[1])
                break
        out.setdefault("channels", {})[modality] = {
            "grid": list(grid),
            "rows": n_rows,
            "probes_per_finger": grid[0] * grid[1],
        }
    return out

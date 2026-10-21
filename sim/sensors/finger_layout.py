"""Procedurally derive tactile probe layouts on a parallel gripper's fingers.

The Panda URDF ships the finger geometry as collision **meshes** (the wrist
links use boxes). Rather than hard-coding pad offsets, we parse the URDF at
runtime and synthesize a probe grid on the face that looks at the other finger
(the grasping surface). Probe frames are expressed in the finger link frame,
which is what Genesis' probe-based tactile sensors expect.

A whole-finger AABB is the wrong pad model: the Panda finger mesh has a raised
flat pad only near the tip (z ~ 0.036..0.054 in the link frame) while the base
and waist are recessed by 2.5..15 mm. Probe rows placed on the AABB face would
float in front of the real surface and close the gripper early with inflated
depth readings. So for mesh collisions we take the mesh **vertices**, select
the pad slab (points within ``pad_tol_m`` of the innermost surface,
``d = v @ inner``), and snap each probe to the local surface height of its own
tangential neighborhood. ``pad_tol_m`` must stay below the recess depth or the
slab merges with the rest of the finger: on the Panda the extracted pad face
is identical for any slab of 0.5..2 mm (28 vertices, 17.4 x 17.5 mm), so the
default 1.5 mm sits in the middle of the stable window.

The closure axis comes from the two collision-shape centers in the shared
parent (hand) frame — NOT from the joint origins, which coincide on mimic
grippers like the Panda's (both finger joints share ``xyz="0 0 0.0584"``), and
NOT from the pad-face centers, whose 0.2 mm separation on a closed gripper is
noise. When no collision geometry is available we fall back to the
joint-origin difference.

Pure XML + numpy: this module imports without Genesis so layouts can be
inspected and unit-tested independently.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


@dataclass
class FingerLayout:
    """Probe grid for one finger link, in the link's local frame."""

    link_name: str
    probe_local_pos: np.ndarray  # (P, 3) float32
    probe_local_normal: np.ndarray  # (P, 3) float32, unit vectors
    pad_box_origin: np.ndarray  # (3,) pad-face center (link frame)
    pad_box_size: np.ndarray  # (3,) pad-face extents; 0 along the inner axis
    inner_dir: np.ndarray  # (3,) unit vector pointing toward the other finger

    @property
    def num_probes(self) -> int:
        return int(self.probe_local_pos.shape[0])

    @property
    def pad_face_area_m2(self) -> float:
        """Area of the pad face (m^2): the product of the non-zero extents.

        ``pad_box_size`` is zero along the inner (contact-normal) axis, so this
        is the grasping face's own area — 305.949 mm^2 on the Panda. Used as the
        single contact area the pad-face policy splits over a finger's contact
        points (docs §9-R37).
        """
        size = np.abs(np.asarray(self.pad_box_size, dtype=np.float64))
        return float(np.prod(size[size > 0.0]))


def _parse_vec(text: str | None, default: tuple[float, ...]) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(v) for v in text.split()], dtype=np.float64)


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """URDF fixed-axis convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    r, p, y = (float(v) for v in rpy)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


@dataclass
class _Box:
    origin: np.ndarray  # (3,)
    size: np.ndarray  # (3,)
    rpy: np.ndarray  # (3,)


def parse_link_boxes(urdf_path: str | Path, links: set[str] | None = None) -> dict[str, list[_Box]]:
    """Map link name -> list of collision shapes (origin/size/rpy in link frame).

    ``<box>`` collisions are read directly; ``<mesh>`` collisions (.obj) are
    reduced to their axis-aligned bounding box in the link frame (collision
    origin applied). ``derive_finger_layout`` prefers the raw vertices from
    ``parse_link_meshes`` — this AABB view is the coarse fallback.
    ``links`` optionally restricts parsing (large URDFs).
    """
    root = ET.parse(urdf_path).getroot()
    out: dict[str, list[_Box]] = {}
    for link in root.findall("link"):
        name = link.get("name")
        if links is not None and name not in links:
            continue
        boxes: list[_Box] = []
        for coll in link.findall("collision"):
            origin_el = coll.find("origin")
            xyz = _parse_vec(origin_el.get("xyz") if origin_el is not None else None, (0.0, 0.0, 0.0))
            rpy = _parse_vec(origin_el.get("rpy") if origin_el is not None else None, (0.0, 0.0, 0.0))
            box = coll.find("geometry/box")
            if box is not None:
                boxes.append(_Box(origin=xyz, size=_parse_vec(box.get("size"), (0.0, 0.0, 0.0)), rpy=rpy))
                continue
            mesh = coll.find("geometry/mesh")
            if mesh is not None:
                verts = _mesh_vertices(urdf_path, mesh, xyz, rpy)
                if verts is not None:
                    lo, hi = verts.min(axis=0), verts.max(axis=0)
                    boxes.append(_Box(origin=(lo + hi) / 2.0, size=hi - lo, rpy=np.zeros(3)))
                continue
        if boxes:
            out[name] = boxes
    return out


def _resolve_mesh_path(urdf_path: str | Path, filename: str) -> Path | None:
    """Resolve a URDF mesh reference (``package://`` or relative) to a file."""
    if not filename:
        return None
    if filename.startswith("package://"):
        rel = Path(filename[len("package://") :])
        base = Path(urdf_path).resolve().parent
        for cand in (base / rel, *(a / rel for a in base.parents)):
            if cand.is_file():
                return cand
        return None
    cand = Path(filename)
    if not cand.is_absolute():
        cand = Path(urdf_path).resolve().parent / cand
    return cand if cand.is_file() else None


def _obj_vertices(path: Path, scale: np.ndarray) -> np.ndarray:
    """(N, 3) vertices of a Wavefront .obj (``v`` lines only; pure numpy)."""
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        parts = line.split()
        if parts and parts[0] == "v" and len(parts) >= 4:
            rows.append([float(x) for x in parts[1:4]])
    if not rows:
        raise ValueError(f"No vertices found in mesh {path}")
    return np.asarray(rows, dtype=np.float64) * scale


def _mesh_vertices(urdf_path: str | Path, mesh_el, xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray | None:
    """Mesh collision vertices in the link frame (collision origin applied), or None."""
    path = _resolve_mesh_path(urdf_path, mesh_el.get("filename") or "")
    if path is None or path.suffix.lower() != ".obj":
        return None
    scale = _parse_vec(mesh_el.get("scale"), (1.0, 1.0, 1.0))
    return _obj_vertices(path, scale) @ rpy_to_matrix(rpy).T + xyz


def parse_link_meshes(urdf_path: str | Path, links: set[str] | None = None) -> dict[str, np.ndarray]:
    """Map link name -> concatenated collision-mesh vertices (.obj, link frame).

    These are the raw points the pad-slab extraction needs; ``parse_link_boxes``
    keeps the coarser AABB view for callers that only need an extent.
    """
    root = ET.parse(urdf_path).getroot()
    out: dict[str, np.ndarray] = {}
    for link in root.findall("link"):
        name = link.get("name")
        if links is not None and name not in links:
            continue
        chunks = []
        for coll in link.findall("collision"):
            origin_el = coll.find("origin")
            xyz = _parse_vec(origin_el.get("xyz") if origin_el is not None else None, (0.0, 0.0, 0.0))
            rpy = _parse_vec(origin_el.get("rpy") if origin_el is not None else None, (0.0, 0.0, 0.0))
            mesh = coll.find("geometry/mesh")
            if mesh is None:
                continue
            verts = _mesh_vertices(urdf_path, mesh, xyz, rpy)
            if verts is not None:
                chunks.append(verts)
        if chunks:
            out[name] = np.concatenate(chunks, axis=0)
    return out


@dataclass
class _Joint:
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray


def parse_child_joints(urdf_path: str | Path) -> dict[str, _Joint]:
    """Map child link name -> its parent joint (origin pose in the parent frame)."""
    root = ET.parse(urdf_path).getroot()
    out: dict[str, _Joint] = {}
    for joint in root.findall("joint"):
        child_el = joint.find("child")
        parent_el = joint.find("parent")
        if child_el is None or parent_el is None:
            continue
        origin_el = joint.find("origin")
        out[child_el.get("link")] = _Joint(
            parent=parent_el.get("link"),
            child=child_el.get("link"),
            xyz=_parse_vec(origin_el.get("xyz") if origin_el is not None else None, (0.0, 0.0, 0.0)),
            rpy=_parse_vec(origin_el.get("rpy") if origin_el is not None else None, (0.0, 0.0, 0.0)),
        )
    return out


def _union_box(boxes: list[_Box]) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned union (link frame) of possibly rotated boxes."""
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for box in boxes:
        rot = rpy_to_matrix(box.rpy)
        half = box.size / 2.0
        corners = np.array(
            [[sx * half[0], sy * half[1], sz * half[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        )
        world = corners @ rot.T + box.origin
        lo = np.minimum(lo, world.min(axis=0))
        hi = np.maximum(hi, world.max(axis=0))
    return (lo + hi) / 2.0, hi - lo


def _snap_to_axis(vec: np.ndarray, *, tol: float = 0.9) -> np.ndarray:
    axis = int(np.argmax(np.abs(vec)))
    if abs(vec[axis]) < tol:
        raise ValueError(f"Inner direction {vec} is not axis-aligned; pass inner_dir_override in link frame.")
    out = np.zeros(3)
    out[axis] = math.copysign(1.0, vec[axis])
    return out


@dataclass
class _PadFace:
    """Grasping face of one finger in its link frame.

    ``center``/``size`` describe the pad plane (``size`` is 0 along the inner
    axis); ``slab`` holds the pad's own mesh vertices (the per-probe surface
    lookup searches only these, so a probe can never snap onto the recessed
    finger body); it is None for box collisions, whose face is flat.
    """

    center: np.ndarray
    size: np.ndarray
    axis: int
    d_max: float
    slab: np.ndarray | None


def _closure_direction(
    center_a: np.ndarray, center_b: np.ndarray, joint_a: _Joint, joint_b: _Joint
) -> tuple[np.ndarray | None, str]:
    """Unit vector a -> b in the parent frame, plus where it came from.

    ``center_*`` must be collision-shape centers (well separated), never pad
    face centers (near-coplanar once the gripper is closed).
    """
    d = np.asarray(center_b, dtype=np.float64) - np.asarray(center_a, dtype=np.float64)
    if np.linalg.norm(d) > 1e-6:
        return d / np.linalg.norm(d), "collision centers"
    d = np.asarray(joint_b.xyz, dtype=np.float64) - np.asarray(joint_a.xyz, dtype=np.float64)
    if np.linalg.norm(d) > 1e-6:
        return d / np.linalg.norm(d), "joint origins (fallback)"
    return None, "none"


def _pad_face(geo: tuple, inner: np.ndarray, pad_tol_m: float) -> _PadFace:
    """Pad plane of one finger along ``inner`` (link frame).

    ``geo`` is ``("box", origin, size)`` or ``("mesh", verts)``. For meshes the
    pad slab is the set of vertices within ``pad_tol_m`` of the innermost
    surface; its tangential extent is the pad, not the whole finger.
    """
    axis = int(np.argmax(np.abs(inner)))
    if abs(inner[axis]) < 0.99:
        raise ValueError(
            f"Pad face math needs an axis-aligned inner direction in the link frame; got {inner}. "
            "Snap it to an axis or fix inner_dir_override."
        )
    tangents = [a for a in range(3) if a != axis]
    if geo[0] == "box":
        origin, size = geo[1], geo[2]
        half = 0.5 * float((size * np.abs(inner)).sum())
        center = origin + inner * half
        face_size = np.asarray(size, dtype=np.float64).copy()
        face_size[axis] = 0.0
        return _PadFace(center=center, size=face_size, axis=axis, d_max=float(center @ inner), slab=None)

    verts = geo[1]
    d = verts @ inner
    d_max = float(d.max())
    slab = verts[d >= d_max - pad_tol_m]
    lo = slab[:, tangents].min(axis=0)
    hi = slab[:, tangents].max(axis=0)
    center = np.zeros(3)  # bbox center: vertex density must not bias the grid
    center[tangents] = 0.5 * (lo + hi)
    center[axis] = d_max * inner[axis]
    size = np.zeros(3)
    size[tangents] = hi - lo
    return _PadFace(center=center, size=size, axis=axis, d_max=d_max, slab=slab)


def _local_surface_d(face: _PadFace, inner: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Surface height (``v @ inner``) under each probe's tangential position.

    Uses the nearest pad-slab vertex in the tangent plane; a flat pad (the
    Panda's) gives ``d_max`` everywhere, a curved one follows the surface.
    Slab membership bounds the result to ``[d_max - pad_tol_m, d_max]``, so a
    probe can never be placed behind the pad plane. Box faces are flat.
    """
    if face.slab is None:
        return np.full(uv.shape[0], face.d_max)
    tangents = [a for a in range(3) if a != face.axis]
    vt = face.slab[:, tangents]
    d = face.slab @ inner
    out = np.empty(uv.shape[0], dtype=np.float64)
    for i, p in enumerate(uv):
        out[i] = d[np.argmin(((vt - p) ** 2).sum(axis=1))]
    return out


def derive_finger_layout(
    urdf_path: str | Path,
    finger_link_names: tuple[str, str],
    *,
    probe_grid: tuple[int, int] = (4, 3),
    probe_offset_m: float = 5e-4,
    margin_frac: float = 0.15,
    pad_tol_m: float = 1.5e-3,
    box_override: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] | None = None,
    inner_dir_override: dict[str, tuple[float, float, float]] | None = None,
    info: dict | None = None,
) -> list[FingerLayout]:
    """Build a probe grid on the inward face of each finger's pad.

    Parameters
    ----------
    finger_link_names : (left_link, right_link)
        Links that face each other. The closure axis is derived from the two
        collision-shape centers in the shared parent (hand) frame; if no
        collision geometry is available it falls back to the finger joint
        origins.
    probe_grid : (n_along_length, n_along_width)
        Probe counts on the pad face; length = longer tangent extent.
    probe_offset_m : probes float this far off the surface (contact onset
        happens slightly before mesh penetration).
    margin_frac : fraction of each pad extent left free at the borders.
    pad_tol_m : pad-slab thickness for mesh collisions (module docstring).
    box_override / inner_dir_override : optional per-link calibration escape
        hatches if the URDF geometry is unusable (set at Gate 1 if needed).
    info : optional dict filled with ``closure_source``, the hand-frame
        collision and pad-face centers, the pad axes and the pad-slab thickness
        (diagnostics for Gate 1).
    """
    names = tuple(finger_link_names)
    meshes_by_link = parse_link_meshes(urdf_path, links=set(names))
    boxes_by_link = parse_link_boxes(urdf_path, links=set(names))
    joints = parse_child_joints(urdf_path)

    ja, jb = (joints.get(name) for name in names)
    if ja is None or jb is None:
        missing = [n for n, j in zip(names, (ja, jb)) if j is None]
        raise ValueError(f"Finger links not found as URDF joint children: {missing}")

    geo: dict[str, tuple] = {}
    for name in names:
        if box_override and name in box_override:
            origin, size = box_override[name]
            geo[name] = ("box", np.asarray(origin, dtype=np.float64), np.asarray(size, dtype=np.float64))
        elif name in meshes_by_link:
            geo[name] = ("mesh", meshes_by_link[name])
        elif name in boxes_by_link:
            origin, size = _union_box(boxes_by_link[name])
            geo[name] = ("box", origin, size)
        else:
            raise ValueError(
                f"No usable collision geometry for {name!r} (box or .obj mesh); pass box_override."
            )

    def _link_inner(name: str, joint: _Joint, inner_hand: np.ndarray) -> np.ndarray:
        if inner_dir_override and name in inner_dir_override:
            v = np.asarray(inner_dir_override[name], dtype=np.float64)
            return v / np.linalg.norm(v)
        rot = rpy_to_matrix(joint.rpy)  # link frame -> parent frame
        return _snap_to_axis(rot.T @ inner_hand)

    def _hand_center(name: str, joint: _Joint, faces_: dict[str, _PadFace]) -> np.ndarray:
        return joint.xyz + rpy_to_matrix(joint.rpy) @ faces_[name].center

    # Closure axis from the two collision-shape centers in the parent (hand)
    # frame. Pad *face* centers must NOT be used here: on a closed gripper the
    # pads are nearly coplanar (Panda: y = -0.0001 vs +0.0001), so their
    # difference is 0.2 mm of noise whose sign points at whichever pad plane
    # happens to sit higher — i.e. away from the other finger. The collision
    # centers are 26 mm apart on the Panda and always well-posed.
    centers_hand = {}
    for name, joint in zip(names, (ja, jb)):
        kind, *rest = geo[name]
        c = rest[0] if kind == "box" else 0.5 * (rest[0].min(axis=0) + rest[0].max(axis=0))
        centers_hand[name] = joint.xyz + rpy_to_matrix(joint.rpy) @ c
    axis_ab, closure_source = _closure_direction(centers_hand[names[0]], centers_hand[names[1]], ja, jb)
    if axis_ab is None:
        raise ValueError(
            "Cannot derive the closure axis: collision centers and finger joint origins coincide. "
            "Pass box_override/inner_dir_override."
        )
    inner_hand = {names[0]: axis_ab, names[1]: -axis_ab}
    inner_link = {n: _link_inner(n, j, inner_hand[n]) for n, j in zip(names, (ja, jb))}
    faces = {n: _pad_face(geo[n], inner_link[n], pad_tol_m) for n in names}

    if info is not None:
        info["closure_source"] = closure_source
        info["pad_tol_m"] = float(pad_tol_m)
        info["centers_hand"] = {n: v.tolist() for n, v in centers_hand.items()}
        info["pad_face_centers_hand"] = {
            n: _hand_center(n, j, faces).tolist() for n, j in zip(names, (ja, jb))
        }
        info["pad_axes_hand"] = {n: inner_hand[n].tolist() for n in names}

    layouts: list[FingerLayout] = []
    for name in names:
        inner = inner_link[name]
        face = faces[name]

        tangents = [a for a in range(3) if a != face.axis]
        # length = longer tangent extent.
        tangents.sort(key=lambda a: face.size[a], reverse=True)
        extents = [face.size[a] for a in tangents]
        counts = [probe_grid[0], probe_grid[1]]
        margin = [margin_frac * e for e in extents]

        coord_lists = []
        for extent, margin_a, n in zip(extents, margin, counts):
            usable = max(extent - 2.0 * margin_a, 0.0)
            if n == 1:
                coord_lists.append(np.zeros(1))
            else:
                coord_lists.append(np.linspace(-usable / 2.0, usable / 2.0, n))
        # grid order: first tangent (length) outer loop, second (width) inner.
        c_len, c_wid = np.meshgrid(coord_lists[0], coord_lists[1], indexing="ij")
        offsets = np.stack([c_len.ravel(), c_wid.ravel()], axis=1)  # (P, 2)
        uv = offsets + face.center[tangents]  # absolute tangent coords on the pad

        d_surface = _local_surface_d(face, inner, uv)
        pos = np.tile(face.center, (offsets.shape[0], 1))
        for k, a in enumerate(tangents):
            pos[:, a] = uv[:, k]
        pos[:, face.axis] = inner[face.axis] * (d_surface + probe_offset_m)

        layouts.append(
            FingerLayout(
                link_name=name,
                probe_local_pos=pos.astype(np.float32),
                probe_local_normal=np.tile(inner.astype(np.float32), (pos.shape[0], 1)),
                pad_box_origin=face.center.astype(np.float32),
                pad_box_size=face.size.astype(np.float32),
                inner_dir=inner.astype(np.float32),
            )
        )
    return layouts


def _pack_layouts(layouts: list[FingerLayout]) -> tuple[dict, dict]:
    """One grid's layouts -> (datasets, attrs) for a `layouts/` group."""
    pos = np.concatenate([lay.probe_local_pos for lay in layouts], axis=0)
    normal = np.concatenate([lay.probe_local_normal for lay in layouts], axis=0)
    link_index = np.concatenate(
        [np.full(lay.num_probes, i, dtype=np.int32) for i, lay in enumerate(layouts)], axis=0
    )
    datasets = {
        "probe_local_pos": pos,
        "probe_local_normal": normal,
        "probe_link_index": link_index,
        "link_names": np.asarray([lay.link_name for lay in layouts]),
    }
    return datasets, {"n_probes_total": int(pos.shape[0]), "n_links": len(layouts)}


def grid_tag(grid: tuple[int, int]) -> str:
    """Group name for one probe grid, e.g. (16, 16) -> 'g16x16'."""
    return f"g{int(grid[0])}x{int(grid[1])}"


def layouts_to_hdf5(layouts: list[FingerLayout], *, attrs: dict | None = None) -> dict:
    """Pack a single grid's layouts (datasets at the root of `layouts/`)."""
    datasets, auto = _pack_layouts(layouts)
    return {"attrs": {**auto, **(attrs or {})}, **datasets}


def layouts_to_hdf5_multi(
    groups: dict[tuple[int, int], list[FingerLayout]],
    *,
    default_grid: tuple[int, int],
    modality_grids: dict[str, tuple[int, int]],
    attrs: dict | None = None,
) -> dict:
    """Pack per-grid layouts, keeping the default grid at the root as an alias.

    Layout (docs §5.2)::

        layouts/                      attrs: n_probes_total, n_links, probe_grid,
        layouts/probe_local_pos ...          modality_grids, world_frame, ...
        layouts/grids/<tag>/          attrs: probe_grid, modalities
        layouts/grids/<tag>/probe_local_pos ...

    The root datasets stay *the default modality's grid* so a reader that knows
    nothing about `modality_grids` still sees exactly the rows it always did;
    the authority on any channel's shape is `modality_grids`.
    """
    datasets, auto = _pack_layouts(groups[default_grid])
    mapping = {m: grid_tag(g) for m, g in modality_grids.items()}
    return {
        "attrs": {
            **auto,
            "probe_grid": np.asarray(default_grid, dtype=np.int32),
            "modality_grids": json.dumps(mapping),
            **(attrs or {}),
        },
        **datasets,
        "grids": {
            grid_tag(g): {
                "attrs": {
                    "probe_grid": np.asarray(g, dtype=np.int32),
                    "modalities": np.asarray(sorted(m for m, gm in modality_grids.items() if gm == g)),
                },
                **_pack_layouts(lays)[0],
            }
            for g, lays in groups.items()
        },
    }


def summarize(layouts: list[FingerLayout]) -> str:
    lines = []
    for lay in layouts:
        lines.append(
            f"{lay.link_name}: {lay.num_probes} probes | pad face {np.round(lay.pad_box_origin, 4).tolist()} "
            f"| extents {np.round(lay.pad_box_size, 4).tolist()} | inner dir {lay.inner_dir.tolist()}"
        )
    return "\n".join(lines)

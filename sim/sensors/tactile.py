"""Tactile sensor assembly for the gripper fingers (Genesis 1.4.1).

Construction follows the verified raw-Genesis path: build a
``gs.sensors.<Class>`` options object and register it with
``scene.add_sensor(...)`` BEFORE ``scene.build()``. Read shapes (verified
against the vendored engine):

    KinematicTaxel   -> NamedTuple(force, torque), each (N[,H], *L, 3)
    ContactDepthProbe-> depth (N[,H], *L)          meters
    ContactProbe     -> bool  (N[,H], *L)
    Contact          -> bool  (N[,H], 1)
    ContactForce     -> force (N[,H], 3)
    ProximityTaxel   -> NamedTuple(force, torque), each (N[,H], *L, 3); force only

``history_length`` is set to the control decimation so one read per policy
step carries every sim substep; a temporal reduction (median, matching
dexterous-hands' proven convention) collapses the H axis before recording.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from sim.sensors.finger_layout import FingerLayout

if TYPE_CHECKING:  # torch is an annotation only; this module must stay numpy-only
    import torch

# ---------------------------------------------------------------------------
# Tactile parameters: the single source of truth (format 1.6).
#
# The YAML only chooses the modalities. Everything else lives here so the
# sensor definition and the data contract cannot drift apart. The two recorded
# signals are:
#
#   force     -> AT-VLA-style 6D resultant per finger: [Fn(3), Ft(3)]
#   elastomer -> 2D marker displacement in the finger pad's tangent plane
#
# `PROBE_GRID["default"]` is the coarse grid behind the 12 force probes per
# finger; `PROBE_GRID["elastomer"]` is the dense 16x16 marker field.
# ---------------------------------------------------------------------------

PROBE_GRID: dict[str, tuple[int, int]] = {"default": (4, 3), "elastomer": (16, 16)}
PROBE_OFFSET_M = 5e-4
PAD_TOL_M = 1.5e-3
PROBE_RADIUS_M = 0.004
PROXIMITY_SAMPLE_POINTS = 500
AS_GRID = True
# HydroShear length scales (1/m^2): 5e4 -> ~4.5 mm normal bulge, 2e4 -> ~7 mm
# shear spread. Do NOT set compressibility < 1 or elastomer_thickness > 0:
# that switches the engine to an FFT path that crashes (docs §9-R8).
ELASTOMER_PARAMS: dict[str, float] = {"lambda_d": 50000.0, "lambda_s": 20000.0}
CONTACT_LINKS = True
TEMPORAL_REDUCTION = "median"
# Format 1.6 records the derived signals in the finger/link frame. The 6D
# decomposition is already frame-consistent; the 2D marker field is, by
# definition, expressed in the pad tangent plane.
WORLD_FRAME = False
FORCE_6D = True
MARKER_2D = True

CHANNEL_FEATURES: dict[str, int] = {"force": 6, "elastomer": 2}
CHANNEL_FRAMES: dict[str, str] = {"force": "link", "elastomer": "pad_tangent"}

VECTOR_MODALITIES = frozenset({"force", "force_torque", "proximity", "elastomer"})
FEATURES_PER_PROBE = {
    "force": 3,
    "force_torque": 6,
    # ProximityTaxel also returns torque, but `_extract` records the force field
    # only (`data.force`), so the recorded channel is 3 wide.
    "proximity": 3,
    "elastomer": 3,
    "depth": 1,
    "bool": 1,
}
# Modalities whose builder needs a non-empty track_link_idx (point-cloud sensors).
# These also consume `probe_local_normal` (their force is aligned with it), unlike
# force/depth, whose normal field is only read by the opt-in raycast backend.
NEEDS_TRACK_LINK = frozenset({"elastomer", "proximity"})


def grid_for(modality: str) -> tuple[int, int]:
    """(ny, nx) probe grid for one modality; falls back to ``default``."""
    return PROBE_GRID.get(modality, PROBE_GRID["default"])


def primary_modality(modalities: Sequence[str]) -> str:
    """Modality whose grid is the file-level `layouts/` alias (docs §5.2)."""
    return modalities[0] if modalities else "force"


def channel_format(modality: str, n_fingers: int, grid: tuple[int, int]) -> dict[str, Any] | None:
    """Declared per-file shape/frame of a derived tactile channel (format 1.6).

    ``force`` is one 6D resultant per finger; ``elastomer`` is one 2D tangent
    displacement per marker. Modalities outside ``CHANNEL_FEATURES`` (legacy
    probe channels) have no derived format and return None.
    """
    features = CHANNEL_FEATURES.get(modality)
    if features is None:
        return None
    if modality == "force":
        shape = [int(n_fingers), features]
    else:
        shape = [int(n_fingers) * int(grid[0]) * int(grid[1]), features]
    return {"features": features, "frame": CHANNEL_FRAMES[modality], "shape": shape}


def pad_tangent_basis(
    probe_local_pos: np.ndarray, inner_dir: np.ndarray, grid: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal pad-tangent basis ``(e0, e1)`` in the finger link frame.

    ``e0`` follows grid axis 0 (pad length), ``e1`` axis 1 (pad width). Both
    are orthogonalized against the pad normal (``n = -inner_dir``, outward) so
    ``disp @ [e0, e1]`` is exactly the in-plane marker displacement. The sign
    of each axis is inherited from the finger's grid, so the pair can be left-
    or right-handed on a given pad; consumers should treat ``(u, v)`` as
    grid-axis coordinates, not as a world-oriented frame.
    """
    ny, nx = int(grid[0]), int(grid[1])
    if ny < 1 or nx < 1:
        raise ValueError(f"probe grid entries must be >= 1, got {grid!r}")
    pos = np.asarray(probe_local_pos, dtype=np.float64).reshape(ny, nx, 3)
    n = -np.asarray(inner_dir, dtype=np.float64)
    n_norm = np.linalg.norm(n)
    if n_norm == 0.0:
        raise ValueError("inner_dir is the zero vector; cannot define the pad normal")
    n = n / n_norm

    def tangent(v: np.ndarray) -> np.ndarray:
        return v - float(v @ n) * n

    e0 = None
    e1 = None
    if ny > 1:
        d0 = tangent(pos[1, 0] - pos[0, 0])
        d0_norm = np.linalg.norm(d0)
        if d0_norm > 0.0:
            e0 = d0 / d0_norm
    if nx > 1:
        d1 = tangent(pos[0, 1] - pos[0, 0])
        if e0 is not None:
            d1 = d1 - float(d1 @ e0) * e0
        d1_norm = np.linalg.norm(d1)
        if d1_norm > 0.0:
            e1 = d1 / d1_norm
    if e0 is None and e1 is not None:
        e0 = np.cross(e1, n)
        e0 /= np.linalg.norm(e0)
    if e1 is None and e0 is not None:
        e1 = np.cross(n, e0)
        e1 /= np.linalg.norm(e1)
    if e0 is None or e1 is None:
        # 1x1 grid: no lattice direction exists; any tangent pair is valid.
        axis = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e0 = tangent(axis)
        e0 /= np.linalg.norm(e0)
        e1 = np.cross(n, e0)
        e1 /= np.linalg.norm(e1)
    return e0, e1


def force_6d(force_link: np.ndarray, inner_dir: np.ndarray) -> np.ndarray:
    """Sum probe forces ``(P,3)`` and decompose into ``[Fn(3), Ft(3)]``.

    ``Fn`` is the component along the outward pad normal ``n = -inner_dir``,
    so a finger being compressed reads positive. ``Ft = F - Fn`` is the
    tangential resultant. This is AT-VLA's 6D force (normal + tangential), not
    force + torque.
    """
    forces = np.asarray(force_link, dtype=np.float64)
    if forces.ndim != 2 or forces.shape[1] != 3:
        raise ValueError(f"force_link must be (P,3), got {forces.shape}")
    f = forces.sum(axis=0)
    n = -np.asarray(inner_dir, dtype=np.float64)
    n_norm = np.linalg.norm(n)
    if n_norm == 0.0:
        raise ValueError("inner_dir is the zero vector; cannot define the pad normal")
    n = n / n_norm
    f_n = float(f @ n) * n
    return np.concatenate([f_n, f - f_n]).astype(np.float32)  # (6,)


def marker_2d(
    disp_link: np.ndarray,
    probe_local_pos: np.ndarray,
    inner_dir: np.ndarray,
    grid: tuple[int, int],
) -> np.ndarray:
    """``(P,3)`` link-frame displacement -> ``(P,2)`` pad-tangent displacement.

    The normal component is intentionally discarded (format 1.6 stores the
    AT-VLA N x 2 marker format). ``u`` follows grid axis 0, ``v`` axis 1.
    """
    e0, e1 = pad_tangent_basis(probe_local_pos, inner_dir, grid)
    disp = np.asarray(disp_link, dtype=np.float64)
    if disp.ndim != 2 or disp.shape[1] != 3:
        raise ValueError(f"disp_link must be (P,3), got {disp.shape}")
    return np.stack([disp @ e0, disp @ e1], axis=1).astype(np.float32)  # (P,2)


@dataclass
class TactileHandle:
    """One live Genesis tactile sensor plus the metadata needed to read it."""

    name: str  # e.g. "tactile_force_panda_leftfinger"
    modality: str
    link_name: str
    sensor: Any
    num_probes: int
    history_length: int
    grid_shape: tuple[int, int] | None = None

    @property
    def features_per_probe(self) -> int:
        return FEATURES_PER_PROBE[self.modality]


def _nested(arr: np.ndarray):
    """Convert an array to nested tuples (pydantic-validated option fields)."""
    if arr.ndim == 1:
        return tuple(float(v) for v in arr)
    return tuple(_nested(a) for a in arr)


def _probe_field(values: np.ndarray, grid: tuple[int, int] | None):
    """Probe positions/normals -> nested tuples, as an (ny,nx,3) grid if given.

    ``grid`` is the *modality's* grid (``grid_for(modality)``), not the
    scene-level one: densities differ per modality (docs §5.2).
    """
    arr = np.asarray(values, dtype=np.float64)
    if grid is not None:
        ny, nx = grid
        arr = arr.reshape(ny, nx, 3)
    return _nested(arr)


def _build_options(
    modality: str,
    *,
    entity_idx: int,
    link_idx_local: int,
    probe_pos,
    probe_radius: float,
    history_length: int,
    track_link_idx: tuple[int, ...],
    probe_normal=None,
    extra: dict[str, Any],
):
    import genesis as gs

    kwargs: dict[str, Any] = dict(
        entity_idx=entity_idx,
        link_idx_local=link_idx_local,
        probe_local_pos=probe_pos,
        probe_radius=probe_radius,
        history_length=history_length,
        **extra,
    )
    if modality in NEEDS_TRACK_LINK:
        if not track_link_idx:
            raise ValueError(f"Tactile modality {modality!r} requires track_link_idx (global link indices).")
        kwargs["track_link_idx"] = tuple(int(i) for i in track_link_idx)
        # Without this the engine's default (0,0,1) — a link-frame direction that
        # is not the pad normal on the Panda fingers — would define the force axis.
        kwargs["probe_local_normal"] = probe_normal

    if modality in ("force", "force_torque"):
        return gs.sensors.KinematicTaxel(**kwargs)
    if modality == "depth":
        return gs.sensors.ContactDepthProbe(**kwargs)
    if modality == "bool":
        return gs.sensors.ContactProbe(**kwargs)
    if modality == "elastomer":
        return gs.sensors.ElastomerTaxel(**kwargs)
    if modality == "proximity":
        return gs.sensors.ProximityTaxel(**kwargs)
    raise ValueError(f"Unknown tactile modality: {modality!r}")


def build_finger_sensors(
    scene,
    *,
    entity,
    layouts_by_modality: dict[str, list[FingerLayout]],
    modalities: Sequence[str],
    decimation: int,
    extra_params: dict[str, dict[str, Any]] | None = None,
    track_link_idx: tuple[int, ...] = (),
) -> list[TactileHandle]:
    """Register one sensor per (modality, finger link). Returns live handles.

    Density is per modality (``grid_for``), so a dense gel marker grid does
    not force the same probe count on force/depth: ``layouts_by_modality`` maps
    each modality to the finger layouts derived for *its* grid.

    Must be called after the robot entity is added and BEFORE scene.build().
    """
    extra_params = extra_params or {}
    for modality in modalities:
        if modality not in layouts_by_modality or not layouts_by_modality[modality]:
            raise ValueError(
                f"No finger layouts for tactile modality {modality!r}; "
                "scene.py must derive one layout set per distinct probe grid."
            )
    handles: list[TactileHandle] = []
    for modality in modalities:
        grid = grid_for(modality)
        for layout in layouts_by_modality[modality]:
            link = entity.get_link(layout.link_name)
            field_grid = grid if AS_GRID else None
            opts = _build_options(
                modality,
                entity_idx=entity.idx,
                link_idx_local=link.idx_local,
                probe_pos=_probe_field(layout.probe_local_pos, field_grid),
                probe_radius=PROBE_RADIUS_M,
                history_length=decimation,
                track_link_idx=track_link_idx,
                probe_normal=_probe_field(layout.probe_local_normal, field_grid),
                extra=dict(extra_params.get(modality, {})),
            )
            sensor = scene.add_sensor(opts)
            handles.append(
                TactileHandle(
                    name=f"tactile_{modality}_{layout.link_name}",
                    modality=modality,
                    link_name=layout.link_name,
                    sensor=sensor,
                    num_probes=layout.num_probes,
                    history_length=decimation,
                    grid_shape=field_grid,
                )
            )
    # read_all() keys readings by handle name, so a collision would silently
    # swap two modalities' tensors rather than fail.
    if len({h.name for h in handles}) != len(handles):
        raise ValueError(f"Duplicate tactile sensor names: {[h.name for h in handles]}")
    return handles


def build_link_contact_sensors(scene, *, entity, link_names: list[str]) -> list[TactileHandle]:
    """Link-level Contact (bool) + ContactForce (N,3) sensors on each link."""
    import genesis as gs

    handles: list[TactileHandle] = []
    for name in link_names:
        link = entity.get_link(name)
        for modality, cls in (("link_bool", gs.sensors.Contact), ("link_force", gs.sensors.ContactForce)):
            sensor = scene.add_sensor(cls(entity_idx=entity.idx, link_idx_local=link.idx_local))
            handles.append(
                TactileHandle(
                    name=f"tactile_{modality}_{name}",
                    modality=modality,
                    link_name=name,
                    sensor=sensor,
                    num_probes=1,
                    history_length=0,
                )
            )
    return handles


# --------------------------------------------------------------------------- read


def _extract(data: Any, modality: str) -> "torch.Tensor":
    """Pull the relevant tensor out of a sensor reading."""
    if modality in ("force", "force_torque", "proximity"):
        force = data.force
        if modality == "force":
            return force
        if modality == "force_torque":
            import torch  # local: keep the module importable without torch

            return torch.cat([force, data.torque], dim=-1)
        return force  # proximity: force field only
    if modality == "elastomer":
        return data
    if isinstance(data, tuple):
        return data[0]
    return data


def temporal_reduce(tensor: "torch.Tensor", history_length: int, mode: str = "median") -> "torch.Tensor":
    """Collapse the within-step history axis (axis 1) of an (N, H, *S) tensor."""
    if history_length <= 0 or tensor.ndim < 3 or tensor.shape[1] != history_length:
        return tensor
    if mode == "median":
        return tensor.float().median(dim=1).values
    if mode == "last":
        return tensor[:, -1]
    if mode == "none":
        moved = tensor.movedim(1, -1)
        if moved.ndim >= 3:
            return moved.reshape(*moved.shape[:-2], moved.shape[-2] * moved.shape[-1])
        return moved
    raise ValueError(f"Unknown temporal reduction: {mode!r}")


def read_all(
    handles: list[TactileHandle],
    *,
    envs_idx: Any = None,
    temporal_reduction: str = "median",
) -> dict[str, "torch.Tensor"]:
    """Read every handle once; returns name -> (N[,P[,F]]) tensor (link frame)."""
    out: dict[str, "torch.Tensor"] = {}
    for h in handles:
        tensor = _extract(h.sensor.read(envs_idx=envs_idx), h.modality)
        tensor = temporal_reduce(tensor, h.history_length, temporal_reduction)
        out[h.name] = tensor
    return out


def rotate_to_world(vec: "torch.Tensor", quat: "torch.Tensor") -> "torch.Tensor":
    """Rotate link-frame vectors (..., 3) into world frame given (N,4) wxyz quats."""
    import genesis.utils.geom as gu

    flat = vec.reshape(vec.shape[0], -1, 3)
    out = gu.transform_by_quat(flat, quat)
    return out.reshape(vec.shape)

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
from typing import Any

import numpy as np
import torch

from tempvla.config import TactileConfig
from tempvla.sensors.finger_layout import FingerLayout

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

    ``grid`` is the *modality's* grid (``cfg.grid_for(modality)``), not the
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
    cfg: TactileConfig,
    decimation: int,
    extra_params: dict[str, dict[str, Any]] | None = None,
    track_link_idx: tuple[int, ...] = (),
) -> list[TactileHandle]:
    """Register one sensor per (modality, finger link). Returns live handles.

    Density is per modality (``cfg.grid_for``), so a dense gel marker grid does
    not force the same probe count on force/depth: ``layouts_by_modality`` maps
    each modality to the finger layouts derived for *its* grid.

    Must be called after the robot entity is added and BEFORE scene.build().
    """
    extra_params = extra_params or {}
    for modality in cfg.modalities:
        if modality not in layouts_by_modality or not layouts_by_modality[modality]:
            raise ValueError(
                f"No finger layouts for tactile modality {modality!r}; "
                "scene.py must derive one layout set per distinct probe grid."
            )
    handles: list[TactileHandle] = []
    for modality in cfg.modalities:
        grid = cfg.grid_for(modality)
        for layout in layouts_by_modality[modality]:
            link = entity.get_link(layout.link_name)
            field_grid = grid if cfg.as_grid else None
            opts = _build_options(
                modality,
                entity_idx=entity.idx,
                link_idx_local=link.idx_local,
                probe_pos=_probe_field(layout.probe_local_pos, field_grid),
                probe_radius=cfg.probe_radius_m,
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


def _extract(data: Any, modality: str) -> torch.Tensor:
    """Pull the relevant tensor out of a sensor reading."""
    if modality in ("force", "force_torque", "proximity"):
        force = data.force
        if modality == "force":
            return force
        torque = data.torque
        if modality == "force_torque":
            return torch.cat([force, torque], dim=-1)
        return force  # proximity: force field only
    if modality == "elastomer":
        return data
    if isinstance(data, tuple):
        return data[0]
    return data


def temporal_reduce(tensor: torch.Tensor, history_length: int, mode: str = "median") -> torch.Tensor:
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
) -> dict[str, torch.Tensor]:
    """Read every handle once; returns name -> (N[,P[,F]]) tensor (link frame)."""
    out: dict[str, torch.Tensor] = {}
    for h in handles:
        tensor = _extract(h.sensor.read(envs_idx=envs_idx), h.modality)
        tensor = temporal_reduce(tensor, h.history_length, temporal_reduction)
        out[h.name] = tensor
    return out


def rotate_to_world(vec: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """Rotate link-frame vectors (..., 3) into world frame given (N,4) wxyz quats."""
    import genesis.utils.geom as gu

    flat = vec.reshape(vec.shape[0], -1, 3)
    out = gu.transform_by_quat(flat, quat)
    return out.reshape(vec.shape)

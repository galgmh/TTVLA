"""TemperatureGrid assembly + per-episode ground-truth seeding (Genesis 1.4.1).

This module is the **single source of truth** for every temperature parameter
the task YAML does not expose. The YAML chooses only the two grid sizes and
the room temperature; everything else — contact-area policy, measured-channel
non-idealities, materials, link-temperature coupling — is a module-level
constant below. A probe that deliberately sweeps one of these overrides the
module constant before ``build_scene()`` (the historical R33/R37 probes did
exactly that; they were removed in the 2026-10-20 scripts cleanup, docs §9-R46).

The recorded channel is the *measured* one (``RECORD_GROUND_TRUTH = False``,
docs §9-R36): the finger sensors are the engine's default lumped
``grid_size=(1,1,1)`` cell plus the official non-idealities (RC lag, noise,
quantization — see :func:`_finger_imperfection_kwargs`); the object grids carry
no non-idealities, so their ``read()`` equals the ground truth.

Verified traps handled here (shared metadata is per sensor CLASS per scene):

* ``ambient_temperature`` / ``convection_coefficient``: last non-None wins —
  we pass identical values to every TemperatureGrid.
* ``simulate_all_link_temperatures``: sticky-true — same value everywhere.
* ``properties_dict``: merged at build; overriding a key afterwards does NOT
  refresh the stored tensor. Per-episode initial temperatures therefore go
  through :func:`seed_temperature` (direct write into the manager's ground
  truth buffer), executed AFTER ``scene.reset()`` and BEFORE the next step.
* Grid geometry: the engine cuts each grid from the link's *world* AABB (two
  diagonal corners inverse-transformed into the link frame), so a multi-cell
  finger grid's pad-normal axis collapses to micrometers and the grid covers
  the whole finger, not the pad (docs §9-R26). Finger grids are the lumped
  (1,1,1) box, where no axis can collapse.
* Entities must still be built at identity pose and oriented afterwards via a
  root DOF (the ``aabb`` box, and the objects' ``object_grid``, are frozen at
  build time): docs §9-R21.
* ``scene.reset()`` clobbers seeded values back to ``base_temperature``;
  ``read()`` / ``read_ground_truth()`` only reflect the seed after one step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch


# ----------------------------------------------------------------- constants
#
# Everything the YAML no longer exposes. The task config only picks the two
# grid sizes and the room temperature; the values below are project-wide.

# The engine's per-voxel radiation/convection term divides a per-m^2 flux by
# (rho*cp*voxel_volume) with no face-area factor, so it over-cools by ~1/dx^2
# (~1e4x at our voxel sizes: a 130 C cylinder reaches ambient in ~0.4 s instead
# of the physical ~1e3 s). Keep it at 0 and pair it with emissivity 0: an
# episode is then adiabatic except for contact conduction, which is the channel
# this task is about. See docs §9-R14.
CONVECTION_COEFFICIENT = 0.0

# Sticky-true across all TemperatureGrid sensors in a scene (shared metadata).
# Required for cross-link coupling: T_other then follows the object's live link
# temperature instead of its material base temperature (docs §9-R17/R32).
SIMULATE_ALL_LINK_TEMPERATURES = True

# Official sensor non-idealities (docs §9-R36). All three are official engine
# options that act on the *measured* channel only (`read()`), never on
# read_ground_truth(); `sensor_time_constant` is stored per sensor (the engine
# concats it per sensor index), so they can be — and are — applied to the
# finger sensors alone: object grids keep 0/0/0 and record the raw field.
# sigma/step follow ~/tactile-genesis's real element; the RC lag is 0.05 s —
# the tau sweep behind §9-R36 showed 0.15 s (~the pad's own contact-coupling
# constant) swallows the metal/non-metal contrast (approach_rate < 2x), while
# 0.05 s keeps 2.00-2.33x per finger with the lag still applied.
SENSOR_TIME_CONSTANT = 0.05  # first-order RC lag, seconds (0 = off)
NOISE = 0.05  # additive white noise std, deg C (0 = off)
RESOLUTION = 0.1  # quantization increment, deg C (0 = off)

# Weight of the engine's penetration-area fallback:
# area = max(contact_polygon_area, pi * weight * penetration). At realistic
# penetrations (mm) the fallback dwarfs the real patch and the explicit contact
# update diverges, so keep it at 0 (docs §9-R16).
CONTACT_DEPTH_WEIGHT = 0.0

# Contact-area policy fed to the engine's contact-heat kernel (docs §9-R37):
#   "engine"   -- historic. Per link-pair group: the 3-D polygon of the
#                 penetrating corners (>= 3 points), else gs.EPS +
#                 sum(pi * penetration) -- a length used as an area. With the
#                 lumped (1,1,1) fingertip cell this makes one grasp's two
#                 fingers differ ~24x between the 2- and 3-point phases
#                 (261 mm^2 vs 43 mm^2 per point, measured; heat ~ A^2, docs
#                 §9-R33/R36).
#   "pad_face" -- sim/compat/sensor_compat.py rewrites every contact that
#                 touches a finger link to A_i = A_face/sqrt(n_i), n_i being
#                 that finger's contact count this substep: sum(A_i^2) =
#                 A_face^2 in every phase, i.e. one pad-sized contact at the
#                 real ~306 mm^2 silicone face. Requires
#                 CONTACT_DEPTH_WEIGHT == 0 (the kernel's max() would
#                 out-vote the rewritten areas).
# Object-object / table contacts keep the engine's areas under both; the
# object grids and the shared link-temperature loop consume whatever the
# finger policy leaves in the buffer. The value is "pad_face" (2026-10-14,
# pre-M9): when it was still a YAML key every shipped config opted in
# explicitly, because a forgotten line must not silently put one file on the
# historic physics while the rest of the dataset is on the physical one. It is
# fixed here now, so no config can be mixed.
CONTACT_AREA_POLICY = "pad_face"

# False = the dataset records read(), the measured channel: the finger
# sensors carry the official non-idealities above (RC lag, noise,
# quantization); the object grids carry none, so their read() equals the
# ground truth bit for bit (docs §9-R36). True = legacy ground-truth
# recording (formats <= 1.4).
RECORD_GROUND_TRUTH = False

# Finger pad + environment materials (base temp = ambient unless seeded).
# k=0.25 is the physical silicone value. It is also the gain of the
# engine's explicit contact update, which must stay stable: c =
# dt*k_eff*A^2/(V^2*rho*cp) < 0.5 per substep, with A the contact patch
# and V = voxel_volume + gs.EPS (the eps term dominates: ~119 mm^3 vs the
# cell's own volume). gate1's `_thermal_diag` asserts c < 0.5 on every
# scene. k sets the material contrast through k_eff = 2*k_pad*k_obj/
# (k_pad+k_obj): at the old 0.02 the contrast was pinned at ~1.03x (docs
# §9-R34); at 0.25 the finger rise is aluminium:ABS ~2.2x (docs §9-R35).
# Do NOT pair high k with the pi*penetration area fallback
# (CONTACT_DEPTH_WEIGHT > 0) — that combination diverges (docs §9-R16).
FINGER_PROPERTIES: dict[str, float] = {
    "conductivity": 0.25,
    "density": 1100.0,
    "specific_heat": 1400.0,
    "emissivity": 0.0,
}
ENVIRONMENT_PROPERTIES: dict[str, float] = {
    "conductivity": 0.15,
    "density": 700.0,
    "specific_heat": 1600.0,
    "emissivity": 0.0,
}


@dataclass
class TemperatureHandle:
    """One live TemperatureGrid sensor plus recording metadata."""

    name: str  # e.g. "temperature_finger_panda_leftfinger" / "temperature_object_0"
    kind: str  # "finger" | "object"
    link_name: str
    sensor: Any
    grid_size: tuple[int, int, int]
    # Object handles only: index into SceneBundle.objects, so a per-episode seed
    # can address one object. None for finger handles.
    object_index: int | None = None

    @property
    def num_cells(self) -> int:
        return int(self.grid_size[0] * self.grid_size[1] * self.grid_size[2])


def properties_from_material(
    props: Mapping[str, float],
    *,
    base_temperature_c: float | None = None,
):
    """Build a gs.sensors.TemperatureProperties from a config material dict."""
    from genesis.options.sensors import TemperatureProperties

    kwargs = {k: float(v) for k, v in props.items() if k in TemperatureProperties._fields}
    if base_temperature_c is not None:
        kwargs["base_temperature"] = float(base_temperature_c)
    return TemperatureProperties(**kwargs)


def make_properties_dict(
    links: Mapping[int, Mapping[str, float] | Any],
    *,
    default: Mapping[str, float] | Any | None = None,
) -> dict[int, Any]:
    """Assemble the properties_dict (global link idx -> TemperatureProperties).

    Key ``-1`` is the fallback for unlisted links (omit to leave them out of
    contact heat transfer entirely).
    """
    out: dict[int, Any] = {}
    for idx, props in links.items():
        out[int(idx)] = props if hasattr(props, "_fields") else properties_from_material(props)
    if default is not None:
        out[-1] = default if hasattr(default, "_fields") else properties_from_material(default)
    return out


def _shared_kwargs(ambient_temp_c: float) -> dict[str, Any]:
    """Options every TemperatureGrid in the scene shares (last-value-wins metadata)."""
    return dict(
        ambient_temperature=ambient_temp_c,
        convection_coefficient=CONVECTION_COEFFICIENT,
        simulate_all_link_temperatures=SIMULATE_ALL_LINK_TEMPERATURES,
        contact_depth_weight=CONTACT_DEPTH_WEIGHT,
    )


def _finger_imperfection_kwargs() -> dict[str, Any]:
    """Measured-channel non-idealities — finger sensors only (docs §9-R36).

    ``sensor_time_constant`` (first-order RC), ``noise`` and ``resolution`` are
    official engine options stored per sensor index (temperature.py concats
    ``sensor_time_const`` per sensor; base_sensor.py allocates the noise /
    resolution columns per sensor), so they can differ across the scene's
    grids: the finger sensors carry them, the object grids keep 0/0/0 and their
    ``read()`` equals ``read_ground_truth()`` bit for bit. All three act on the
    measured channel only.
    """
    return dict(
        sensor_time_constant=SENSOR_TIME_CONSTANT,
        noise=NOISE,
        resolution=RESOLUTION,
    )


def build_temperature_sensor(
    scene,
    *,
    entity,
    link_idx_local: int,
    link_name: str,
    kind: str,
    grid_size: tuple[int, int, int],
    properties_dict: dict[int, Any],
    ambient_temp_c: float,
    name: str | None = None,
    object_index: int | None = None,
) -> TemperatureHandle:
    """Register one TemperatureGrid. Call BEFORE scene.build().

    ``kind`` also decides the non-ideality options: finger sensors get the
    measured-channel RC/noise/quantization (``_finger_imperfection_kwargs``),
    object grids none — see docs §9-R36.
    """
    import genesis as gs

    opts = gs.sensors.TemperatureGrid(
        entity_idx=entity.idx,
        link_idx_local=int(link_idx_local),
        grid_size=tuple(int(v) for v in grid_size),
        properties_dict=properties_dict,
        **_shared_kwargs(ambient_temp_c),
        **(_finger_imperfection_kwargs() if kind == "finger" else {}),
    )
    sensor = scene.add_sensor(opts)
    return TemperatureHandle(
        name=name or f"temperature_{kind}_{link_name}",
        kind=kind,
        link_name=link_name,
        sensor=sensor,
        grid_size=tuple(int(v) for v in grid_size),
        object_index=object_index,
    )


def build_finger_temperature_sensors(
    scene,
    *,
    entity,
    finger_grid: tuple[int, int, int],
    ambient_temp_c: float,
    finger_link_names: list[str],
    properties_dict: dict[int, Any],
) -> list[TemperatureHandle]:
    """One TemperatureGrid per finger link.

    The lumped (1,1,1) grid — the engine default, one measured temperature per
    fingertip with the official non-idealities (docs §9-R36).
    """
    handles = []
    for link_name in finger_link_names:
        link = entity.get_link(link_name)
        handles.append(
            build_temperature_sensor(
                scene,
                entity=entity,
                link_idx_local=link.idx_local,
                link_name=link_name,
                kind="finger",
                grid_size=finger_grid,
                properties_dict=properties_dict,
                ambient_temp_c=ambient_temp_c,
            )
        )
    return handles


def build_object_temperature_sensor(
    scene,
    *,
    entity,
    object_grid: tuple[int, int, int],
    ambient_temp_c: float,
    properties_dict: dict[int, Any],
    link_idx_local: int = 0,
    link_name: str = "object",
    name: str | None = None,
    object_index: int | None = None,
) -> TemperatureHandle:
    """One TemperatureGrid on an object body.

    ``name`` defaults to the historic ``temperature_object`` for a single-object
    scene; multi-object scenes pass ``temperature_object_<i>`` (that name is an
    in-process handle key, not an HDF5 channel — the recorded channel names under
    ``obs/temperature/`` are unchanged).
    """
    return build_temperature_sensor(
        scene,
        entity=entity,
        link_idx_local=link_idx_local,
        link_name=link_name,
        kind="object",
        grid_size=object_grid,
        properties_dict=properties_dict,
        ambient_temp_c=ambient_temp_c,
        name=name or "temperature_object",
        object_index=object_index,
    )


# ------------------------------------------------------------------ GT seeding


def seed_temperature(sensor, temperatures_c, *, envs_idx=None) -> None:
    """Overwrite every grid cell of one TemperatureGrid with per-env values.

    No public setter exists; this writes the manager's ground-truth buffer,
    which is the state the physics integrator evolves. Order of operations:
    ``scene.reset()`` -> ``seed_temperature(...)`` -> ``scene.step()``.

    temperatures_c: scalar or sequence of length ``n_envs``, in Celsius.
    """
    manager = getattr(sensor, "_manager", None)
    if manager is None or not hasattr(manager, "_ground_truth_intermediate_cache"):
        raise RuntimeError(
            "TemperatureGrid internals changed (sensor._manager._ground_truth_intermediate_cache missing); "
            "re-verify against the installed Genesis version."
        )
    cls = type(sensor)
    dtype = cls._get_intermediate_dtype()
    cls_slice = manager._cache_slices_by_type[cls]
    gt = manager._ground_truth_intermediate_cache[dtype]  # (total_cols, B)

    size = int(sensor._cache_size)
    start = int(cls_slice.start) + int(sensor._cache_offset)
    assert size == int(sensor._shared_metadata.cache_sizes[sensor._idx]), "cache slice mismatch"

    if envs_idx is None:
        envs = torch.arange(gt.shape[1], device=gt.device)
    else:
        envs = manager._sim._scene._sanitize_envs_idx(envs_idx)

    vals = torch.as_tensor(temperatures_c, dtype=gt.dtype, device=gt.device).reshape(-1)
    if vals.numel() == 1:
        vals = vals.expand(envs.numel())
    if vals.numel() != envs.numel():
        raise ValueError(f"temperatures_c has {vals.numel()} entries, need {envs.numel()}")

    gt[start : start + size, envs] = vals.view(1, -1).expand(size, -1)

    md = sensor._shared_metadata
    if md.link_temps.numel() > 0:
        md.link_temps[envs, md.links_idx[sensor._idx]] = vals


def seed_all(handles: list[TemperatureHandle], *, object_temp_c: float | None = None, envs_idx=None) -> None:
    """Convenience: seed the object sensor(s); fingers stay at base temperature."""
    if object_temp_c is None:
        return
    for h in handles:
        if h.kind == "object":
            seed_temperature(h.sensor, object_temp_c, envs_idx=envs_idx)


# ------------------------------------------------------------------- reading


def read_temperature(
    handles: list[TemperatureHandle],
    *,
    ground_truth: bool = False,
    envs_idx=None,
) -> dict[str, torch.Tensor]:
    """Read each grid: name -> (n_envs, *grid_size) Celsius tensors."""
    out: dict[str, torch.Tensor] = {}
    for h in handles:
        if ground_truth:
            out[h.name] = h.sensor.read_ground_truth(envs_idx=envs_idx)
        else:
            out[h.name] = h.sensor.read(envs_idx=envs_idx)
    return out


def read_link_temperatures(handle: TemperatureHandle) -> torch.Tensor | None:
    """Shared per-link temperature table (n_envs, n_links), or None if disabled."""
    temps = handle.sensor.link_temperatures
    return temps if temps.numel() > 0 else None

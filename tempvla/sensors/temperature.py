"""TemperatureGrid assembly + per-episode ground-truth seeding (Genesis 1.4.1).

Verified traps handled here (shared metadata is per sensor CLASS per scene):

* ``ambient_temperature`` / ``convection_coefficient``: last non-None wins —
  we pass identical values to every TemperatureGrid.
* ``simulate_all_link_temperatures``: sticky-true — same value everywhere.
* ``properties_dict``: merged at build; overriding a key afterwards does NOT
  refresh the stored tensor. Per-episode initial temperatures therefore go
  through :func:`seed_temperature` (direct write into the manager's ground
  truth buffer), executed AFTER ``scene.reset()`` and BEFORE the next step.
* The link AABB / voxel volume is frozen at build time -> entities must be
  built at identity pose and oriented afterwards via a root DOF.
* ``scene.reset()`` clobbers seeded values back to ``base_temperature``;
  ``read()`` / ``read_ground_truth()`` only reflect the seed after one step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from tempvla.config import TemperatureConfig


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


def _shared_kwargs(cfg: TemperatureConfig) -> dict[str, Any]:
    return dict(
        ambient_temperature=cfg.ambient_temp_c,
        convection_coefficient=cfg.convection_coefficient,
        simulate_all_link_temperatures=cfg.simulate_all_link_temperatures,
        sensor_time_constant=cfg.sensor_time_constant,
        contact_depth_weight=cfg.contact_depth_weight,
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
    cfg: TemperatureConfig,
    name: str | None = None,
    object_index: int | None = None,
) -> TemperatureHandle:
    """Register one TemperatureGrid. Call BEFORE scene.build()."""
    import genesis as gs

    opts = gs.sensors.TemperatureGrid(
        entity_idx=entity.idx,
        link_idx_local=int(link_idx_local),
        grid_size=tuple(int(v) for v in grid_size),
        properties_dict=properties_dict,
        **_shared_kwargs(cfg),
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
    cfg: TemperatureConfig,
    finger_link_names: list[str],
    properties_dict: dict[int, Any],
) -> list[TemperatureHandle]:
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
                grid_size=cfg.finger_grid,
                properties_dict=properties_dict,
                cfg=cfg,
            )
        )
    return handles


def build_object_temperature_sensor(
    scene,
    *,
    entity,
    cfg: TemperatureConfig,
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
        grid_size=cfg.object_grid,
        properties_dict=properties_dict,
        cfg=cfg,
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

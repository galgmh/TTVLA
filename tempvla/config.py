"""Typed task configuration: dataclasses + YAML loader.

One YAML file describes one task end to end (robot, table, object, sensors,
thermal parameters, randomization ranges, success criteria, recording). The
loader builds nested dataclasses so every module consumes typed config, never
raw dicts.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml


@dataclass
class RobotConfig:
    urdf: str = "urdf/panda_bullet/panda.urdf"
    merge_fixed_links: bool = False  # keep panda_hand / finger links addressable
    arm_joint_names: tuple[str, ...] = (
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    )
    finger_joint_names: tuple[str, ...] = ("panda_finger_joint1", "panda_finger_joint2")
    finger_link_names: tuple[str, ...] = ("panda_leftfinger", "panda_rightfinger")
    hand_link_name: str = "panda_hand"
    # IK target link and hand->grip-center offset (calibrated at Gate 1).
    ik_link_name: str = "panda_hand"
    tcp_offset: tuple[float, float, float] = (0.0, 0.0, 0.105)
    open_finger_pos: float = 0.04
    closed_finger_pos: float = 0.0
    base_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # wxyz
    # Franka "ready" pose + open gripper (build/reset state).
    init_qpos: tuple[float, ...] = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04)
    # PD gains / force limits from the Genesis control_franka example (per dof).
    arm_kp: tuple[float, ...] = (4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0)
    arm_kv: tuple[float, ...] = (450.0, 450.0, 350.0, 350.0, 200.0, 200.0, 200.0)
    finger_kp: tuple[float, ...] = (100.0, 100.0)
    finger_kv: tuple[float, ...] = (10.0, 10.0)
    arm_force_min: tuple[float, ...] = (-87.0, -87.0, -87.0, -87.0, -12.0, -12.0, -12.0)
    arm_force_max: tuple[float, ...] = (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)
    finger_force_min: tuple[float, ...] = (-100.0, -100.0)
    finger_force_max: tuple[float, ...] = (100.0, 100.0)


@dataclass
class ControllerConfig:
    pos_scale_m: float = 0.05
    rot_scale_rad: float = 0.5
    ik_damping: float = 0.1
    ik_max_iters: int = 25
    ik_max_samples: int = 4
    control_hz: float = 20.0
    sim_dt: float = 1.0 / 200.0
    decimation: int = 10
    # Gripper rate limit: max finger-joint travel per second while the target
    # slews toward the commanded position. 0 = unlimited (jump straight to the
    # command, the pre-2026-10-03 behaviour). Full stroke is 0.04 m, so e.g.
    # 0.04 m/s takes 1 s to close. NOTE: this makes the physical approach
    # gentler but does NOT smooth the recorded tactile onset — that 0 -> ~45 N
    # step is probe standoff + the contact-gated SDF query, measured unchanged
    # at 10x slower closing (docs §9-R22).
    finger_speed_m_per_s: float = 0.0


@dataclass
class ExpertConfig:
    """Scripted-expert knobs. Tunable from YAML and by scripts/autotune.py."""

    approach_clearance_m: float = 0.08
    grasp_z_offset_m: float = 0.0  # TCP target z relative to the object center
    pos_tol_m: float = 0.01
    rot_tol_rad: float = 0.2
    descend_tol_m: float = 0.015
    contact_depth_m: float = 1e-5  # probe depth above which contact counts
    squeeze_steps: int = 10  # closed-loop steps kept before lifting
    close_timeout_steps: int = 40  # give up closing after this many steps
    lift_height_m: float = 0.10
    lift_tol_m: float = 0.012
    # Per-episode randomization ranges (widen = more diverse, harder demos).
    grasp_xy_jitter_m: float = 0.004
    approach_yaw_jitter_rad: float = 0.15
    close_extra: float = 1.0  # grip command in CLOSE/LIFT/HOLD (1.0 = fully closed)


def _as_grid(value: Any) -> tuple[int, int]:
    """Coerce one (ny, nx) grid from YAML (list) or Python (tuple)."""
    ny, nx = (int(v) for v in value)
    if ny < 1 or nx < 1:
        raise ValueError(f"probe grid entries must be >= 1, got {value!r}")
    return (ny, nx)


@dataclass
class HeightMapConfig:
    """Dense height-map channel synthesized from the marker field (format 1.3).

    This project's gel sensor is virtual (SDF queries + mesh sampling), so no
    deformable geometry exists for a camera to render. ``obs/tactile/height_map``
    instead resamples the recorded marker field onto an (H, W) image per finger
    with an interpolating spline — semantically the same quantity as UniVTAC's
    rendered ``press_depth``, obtained without a camera (docs §5.2 / §9-R25).
    The map is a pure resampling: it passes through the marker samples and adds
    no information beyond the elastomer channel.

    size : (rows, cols); rows follow probe-grid axis 0, cols axis 1. The output
        spans exactly the marker grid's extent (endpoints aligned). Unit is
        metres, like every other length channel.
    interpolate : ``cubic`` (order-3 spline, clamped to the sample range) or
        ``linear`` (no overshoot by construction).
    """

    enabled: bool = False
    size: tuple[int, int] = (64, 64)
    interpolate: str = "cubic"

    def __post_init__(self) -> None:
        self.size = _as_grid(self.size)
        if self.interpolate not in ("cubic", "linear"):
            raise ValueError(f"height_map.interpolate must be 'cubic' or 'linear', got {self.interpolate!r}")


@dataclass
class TactileConfig:
    modalities: tuple[str, ...] = ("force", "depth")
    imperfection: str = "clean"
    # Probe sampling density, either one grid shared by every modality (legacy
    # form) or per-modality overrides:
    #     probe_grid: [4, 3]                                  # all modalities
    #     probe_grid: {default: [4, 3], elastomer: [16, 16]}  # default + override
    # A dense gel marker grid must NOT be forced onto force/depth: the recorded
    # row count and the obs reshape are per modality (docs §5.2 / §9-R8).
    probe_grid: tuple[int, int] | dict[str, tuple[int, int]] = (4, 3)
    probe_offset_m: float = 5e-4  # probes floated above the pad surface
    pad_tol_m: float = 1.5e-3  # mesh pad-slab thickness: vertices within this of the
    # innermost surface define the pad face (must stay below the finger's recess
    # depth; stable over 0.5..2 mm on the Panda — see sensors/finger_layout.py)
    probe_radius_m: float = 0.004
    # Point-cloud modalities (`proximity`, `elastomer`) only: FPS points sampled
    # per tracked link mesh. Sets the ramp granularity — the reading is a sum over
    # the sampled points inside the probe sphere, so the first nonzero sample
    # appears ~one point-spacing late and the curve steps as points enter. 500 is
    # the engine default (≈4.6 mm spacing on our cylinders); denser = smoother.
    proximity_sample_points: int = 500
    as_grid: bool = True  # pass probe_local_pos as (ny,nx,3) grid (enables crosstalk/FFT paths)
    # Extra kwargs forwarded verbatim to gs.sensors.ElastomerTaxel (the gel-sensor
    # modality): lambda_d / lambda_s (marker-decay length scale, 1/m^2 — the engine
    # default 700 spreads the response over the whole pad), shear_scale,
    # normal_exponent, ... Empty = engine defaults. Do NOT set compressibility<1 or
    # elastomer_thickness>0: that switches the engine to an FFT dilation path that
    # raises `UNSUPPORTED DTYPE: complex` on this build (docs §9-R8).
    elastomer_params: dict[str, float] = field(default_factory=dict)
    contact_links: bool = True  # link-level Contact / ContactForce sensors
    temporal_reduction: str = "median"  # median | last | none (within-step substeps)
    world_frame: bool = True  # rotate vector modalities into world frame when recording
    # Synthesized dense height map under obs/tactile/height_map (format 1.3).
    # Off by default: configs that do not ask for it record exactly what they
    # did before, byte for byte.
    height_map: HeightMapConfig = field(default_factory=HeightMapConfig)

    def __post_init__(self) -> None:
        # YAML delivers lists and `_build` only coerces them when the annotation
        # is a bare tuple — the union above defeats that check, so a legacy
        # `probe_grid: [4, 3]` would otherwise reach the sensors as a list.
        # Normalize once, here, since `grid_for` is the only read point.
        if isinstance(self.probe_grid, dict):
            grids = {str(k): _as_grid(v) for k, v in self.probe_grid.items()}
            grids.setdefault("default", (4, 3))
            self.probe_grid = grids
        else:
            self.probe_grid = _as_grid(self.probe_grid)

    def grid_for(self, modality: str) -> tuple[int, int]:
        """(ny, nx) probe grid for one modality; falls back to ``default``."""
        if isinstance(self.probe_grid, dict):
            return self.probe_grid.get(modality, self.probe_grid["default"])
        return self.probe_grid

    @property
    def primary_modality(self) -> str:
        """Modality whose grid is the file-level `layouts/` alias (docs §5.2)."""
        return self.modalities[0] if self.modalities else "force"


@dataclass
class TemperatureConfig:
    finger_grid: tuple[int, int, int] = (4, 4, 1)
    object_grid: tuple[int, int, int] = (4, 4, 4)
    ambient_temp_c: float = 21.0
    # The engine's per-voxel radiation/convection term divides a per-m^2 flux
    # by (rho*cp*voxel_volume) with no face-area factor, so it over-cools by
    # ~1/dx^2 (~1e4x at our voxel sizes: a 130 C cylinder reaches ambient in
    # ~0.4 s instead of the physical ~1e3 s). Keep it at 0 and pair it with
    # emissivity 0: an episode is then adiabatic except for contact
    # conduction, which is the channel this task is about. See docs §9-R14.
    convection_coefficient: float = 0.0
    record_ground_truth: bool = True  # record read_ground_truth() instead of read()
    # Sticky-true across all TemperatureGrid sensors in a scene (shared metadata).
    simulate_all_link_temperatures: bool = True
    sensor_time_constant: float = 0.0  # RC filter, seconds (0 = off)
    # Weight of the engine's penetration-area fallback:
    # area = max(contact_polygon_area, pi * weight * penetration). At realistic
    # penetrations (mm) the fallback dwarfs the real patch and the explicit
    # contact update diverges, so keep it at 0 (docs §9-R16).
    contact_depth_weight: float = 0.0
    # Finger pad + environment materials (base temp = ambient unless seeded).
    # k is also the gain of the engine's explicit contact update, which must
    # stay stable: c = dt*k_eff*A^2/(V^2*rho*cp) < 0.5 per substep, with A the
    # contact patch and V = voxel_volume + gs.EPS. The pad's voxel_volume is
    # 1.44 mm^3 but gs.EPS = 1.19e-7 m^3 (= float32 eps) dominates it, so V is
    # ~121 mm^3; with the ~2.3 cm^2 patch of a closed grasp, k=0.02 gives
    # c ~ 5e-3 (measured: pads ramp ~+0.07 C/substep, ~2 s to the object
    # temperature). k=10 with the pi*penetration area fallback diverges.
    # See docs §9-R16.
    finger_properties: dict[str, float] = field(
        default_factory=lambda: {
            "conductivity": 0.02,
            "density": 1100.0,
            "specific_heat": 1400.0,
            "emissivity": 0.0,
        }
    )
    environment_properties: dict[str, float] = field(
        default_factory=lambda: {
            "conductivity": 0.15,
            "density": 700.0,
            "specific_heat": 1600.0,
            "emissivity": 0.0,
        }
    )


@dataclass
class CamerasConfig:
    """Optional camera channels (Gate 3).

    ``agentview`` is a fixed third-person camera; ``eye_in_hand`` is attached to
    the hand link with a constant offset, so its pose is expressed in the *hand*
    frame (``pos``/``lookat``/``up``, same convention as ``scene.add_camera``).
    Both must share a resolution: BatchRenderer requires identical resolutions
    across cameras.
    """

    enabled: bool = False
    renderer: str = "batch"  # batch | raster | raytrace
    resolution: tuple[int, int] = (128, 128)
    # Close over-the-shoulder view of the workspace: at 128² the object is only
    # a few pixels wide from far away, so the camera sits ~0.55 m from the grasp
    # point (object ≈ 10 px). See docs/architecture.md Gate 3.
    agentview_pos: tuple[float, float, float] = (0.78, -0.38, 0.70)
    agentview_lookat: tuple[float, float, float] = (0.40, 0.0, 0.48)
    fov: float = 45.0
    # Eye-in-hand is mounted on the side of the hand: panda_hand's own mesh
    # spans z -0.026..0.066 in the hand frame, so anything behind the flange
    # looks straight into the hand. The hand is thin in x (|x| < 0.032), so the
    # camera sits out on +x, level with the flange, aimed at the TCP between
    # the fingers (which run z 0.058..0.112). Chosen by looking, not by
    # arithmetic: scripts/probe_camera_pose.py renders a contact sheet, and this
    # pose puts the object at ~21% of the frame with the gripper legible.
    eye_in_hand: bool = True
    eye_in_hand_pos: tuple[float, float, float] = (0.10, 0.0, 0.02)
    eye_in_hand_lookat: tuple[float, float, float] = (0.0, 0.0, 0.10)  # TCP
    # -z so image-up is world-up when the gripper points down: the object sits
    # below the gripper in frame, as a real wrist camera sees it.
    eye_in_hand_up: tuple[float, float, float] = (0.0, 0.0, -1.0)
    eye_in_hand_fov: float = 60.0
    # The default near plane (0.1 m) would clip the fingers of a wrist camera.
    near: float = 0.01
    far: float = 20.0


@dataclass
class ObjectConfig:
    shape: str = "cylinder"  # cylinder | box (box: radius_m is the half-width)
    radius_m: tuple[float, float] = (0.015, 0.025)
    height_m: tuple[float, float] = (0.05, 0.08)
    mass_kg: tuple[float, float] = (0.05, 0.15)
    temp_c: tuple[float, float] = (60.0, 200.0)
    xy_jitter_m: float = 0.04
    yaw_jitter_rad: float = 3.14159
    # Slot on the table for multi-object tasks (table center + place_xy); ignored
    # for the legacy single `object` block, whose canonical slot is the center.
    place_xy: tuple[float, float] = (0.0, 0.0)


@dataclass
class TableConfig:
    pos: tuple[float, float, float] = (0.4, 0.0, 0.0)  # xy of table center
    size: tuple[float, float, float] = (0.6, 0.8, 0.04)
    height_m: float = 0.4  # top surface z
    # Render-only surface color (RGB 0-1, or RGBA). None keeps the engine
    # default `gs.surfaces.Default()` (grey/white) — that is what the v1/v2
    # datasets were shot with, so the default must not change (docs §5.3).
    color: tuple[float, float, float] | None = None


@dataclass
class RecordingConfig:
    output_dir: str = "data"
    keep_failed: bool = True
    compression: str = "gzip"
    compression_opts: int = 4
    contact_burst: bool = True
    burst_pre_s: float = 0.05
    burst_post_s: float = 0.25
    burst_max_windows: int = 10
    # Modalities packed into the 200 Hz burst windows; None = every modality
    # (the historic behavior). A 16x16 marker field is ~218 KiB *per window*
    # gzipped, so this is the knob that keeps burst size bounded (docs §5.5).
    burst_modalities: tuple[str, ...] | None = None


@dataclass
class TaskConfig:
    task_id: str = "pick_hot_cylinder"
    language_template: str = "pick up the hot {material} cylinder"
    materials: tuple[str, ...] = ("aluminum", "copper", "abs")
    # material -> thermal properties, passed to the temperature sensor; keys
    # are reconciled with Genesis' properties_dict at Gate 2.
    material_properties: dict[str, dict[str, float]] = field(default_factory=dict)
    robot: RobotConfig = field(default_factory=RobotConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    expert: ExpertConfig = field(default_factory=ExpertConfig)
    tactile: TactileConfig = field(default_factory=TactileConfig)
    temperature: TemperatureConfig = field(default_factory=TemperatureConfig)
    cameras: CamerasConfig = field(default_factory=CamerasConfig)
    object: ObjectConfig = field(default_factory=ObjectConfig)
    # Multi-object tasks: one ObjectConfig per body, each with its own table slot
    # (`place_xy`) and size/temperature ranges. Empty = the legacy single-object
    # mode, byte-identical to before this field existed (`object` above is used
    # as the only entry). One material still covers the whole file.
    objects: tuple[ObjectConfig, ...] = ()
    # Minimum surface separation between any two objects' table slots; a sampled
    # episode that violates it is re-drawn (see scene.sample_episode). A guard,
    # not a placement: choose place_xy far enough apart that it never fires.
    min_object_gap_m: float = 0.02
    table: TableConfig = field(default_factory=TableConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    max_episode_steps: int = 400
    lift_success_m: float = 0.05
    stable_steps: int = 10
    seed: int = 0

    def object_list(self) -> list[ObjectConfig]:
        """The objects of this task: ``objects`` if given, else ``[object]``."""
        return list(self.objects) if self.objects else [self.object]

    @property
    def steps_per_sec(self) -> float:
        return self.controller.control_hz


def _build(cls: type, data: dict[str, Any] | None) -> Any:
    """Recursively build nested dataclasses from a plain dict (YAML)."""
    if data is None:
        return cls()
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        hint = hints[f.name]
        if dataclasses.is_dataclass(hint):
            kwargs[f.name] = _build(hint, value)
        elif get_origin(hint) is tuple and isinstance(value, list):
            # `tuple[SomeDataclass, ...]` (e.g. TaskConfig.objects): each YAML
            # mapping is built into its own dataclass. Plain tuples keep the
            # historical list->tuple coercion.
            args = [a for a in get_args(hint) if a is not Ellipsis]
            if len(args) == 1 and dataclasses.is_dataclass(args[0]):
                kwargs[f.name] = tuple(_build(args[0], item) for item in value)
            else:
                kwargs[f.name] = tuple(value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def load_task_config(path: str | Path) -> TaskConfig:
    """Load one task YAML into a fully-populated TaskConfig."""
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Task config must be a YAML mapping: {path}")
    return _build(TaskConfig, raw)


def _to_plain(value: Any) -> Any:
    """Nested dataclasses / tuples / dicts -> plain YAML-friendly Python."""
    if dataclasses.is_dataclass(value):
        return {f.name: _to_plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    if isinstance(value, float):
        return round(value, 6)
    return value


def save_task_config(cfg: TaskConfig, path: str | Path, *, extra: dict[str, Any] | None = None) -> None:
    """Write a TaskConfig back to YAML (used by scripts/autotune.py).

    ``extra`` adds top-level keys that are not part of TaskConfig (e.g. a
    ``tuning:`` provenance block); the loader ignores unknown keys.
    """
    payload = _to_plain(cfg)
    if extra:
        payload.update(_to_plain(extra))
    Path(path).write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=100))

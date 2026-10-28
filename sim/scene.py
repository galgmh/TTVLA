"""
Scene construction: Franka Panda + parallel gripper, table, objects, sensors.

Contract highlights (verified against Genesis 1.4.1):

* The robot is built at its authored pose;
  the objects are built at a canonical pose and repositioned per episode.

* All TemperatureGrid sensors in a scene receive identical shared-metadata values.

* Per-episode object temperature set after scene reset before first step.

* Object size/mass and material are sampled once per scene build;
  pose/temperature are sampled per episode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from sim.compat import (
    install_contact_area_policy,
    install_sensor_reset_fix,
    install_texture_resolver_policy,
)
from sim.config import TaskConfig, tray_floor_z, tray_rim_z
from sim.sensors import (
    FingerLayout,
    TactileHandle,
    TemperatureHandle,
    build_finger_sensors,
    build_finger_temperature_sensors,
    build_link_contact_sensors,
    build_object_temperature_sensor,
    derive_finger_layout,
    make_properties_dict,
    properties_from_material,
    seed_temperature,
)
from sim.sensors.tactile import (
    CONTACT_LINKS,
    ELASTOMER_PARAMS,
    PAD_TOL_M,
    PROBE_OFFSET_M,
    PROXIMITY_SAMPLE_POINTS,
    grid_for,
    primary_modality,
)
from sim.sensors.temperature import (
    ENVIRONMENT_PROPERTIES,
    FINGER_PROPERTIES,
)

ASSETS_DIR = Path(__file__).resolve().parent / "assets"


# sampling


@dataclass
class ObjectSpec:
    """One object's frozen geometry/mass (sampled once per scene build).

    ``radius_m`` is the cylinder radius or the box half-width;
    the box morph is ``(2r, 2r, h)``. density (``mass_kg / volume``).
    ``aabb_min``/``aabb_max`` are the built geometry's body-frame AABB in metres.
    ``volume_override_m3`` carries a mesh's measured volume for the density (a primitive's is analytic).
    """

    material: str
    radius_m: float
    height_m: float
    mass_kg: float
    shape: str = "cylinder"
    aabb_min: tuple[float, float, float] | None = None
    aabb_max: tuple[float, float, float] | None = None
    volume_override_m3: float | None = None
    # The body's own orientation, wxyz (None = identity, every primitive).
    base_quat: tuple[float, float, float, float] | None = None

    @property
    def local_aabb(self) -> tuple[np.ndarray, np.ndarray]:
        """Body-frame AABB (lo, hi); analytic for a primitive, measured for a mesh."""
        if self.aabb_min is not None and self.aabb_max is not None:
            return np.asarray(self.aabb_min, dtype=np.float64), np.asarray(self.aabb_max, dtype=np.float64)
        half = 0.5 * float(self.height_m)
        r = float(self.radius_m)
        return np.array([-r, -r, -half]), np.array([r, r, half])

    @property
    def volume_m3(self) -> float:
        if self.volume_override_m3 is not None:
            return float(self.volume_override_m3)
        # A cube keeps the box formula
        if self.shape in ("box", "cube"):
            return float((2.0 * self.radius_m) ** 2 * self.height_m)
        return float(np.pi * self.radius_m**2 * self.height_m)


def sample_object_spec(
    object_cfg: dict[str, Any], material: str, rng: np.random.Generator
) -> ObjectSpec:
    """Sample one object's frozen geometry/mass from its ``objects[i]`` block.

    For ``shape: mesh`` the config's ``radius_m``/``height_m`` ranges are only
    drawn (so the rng stream is the same shape as for a primitive) and then
    replaced by the measured AABB: the built body is the authority on its own
    size, and every downstream reader (placement, metadata, the slot guard's
    neighbours) must see the size that is actually on the table.

    ``shape: cube`` is the one shape whose size is a *single* draw (``side_m``):
    a cube needs its two box components to agree, and two independent uniform
    draws cannot promise that. It is the only branch that samples a different
    number of values.
    """
    o = object_cfg
    shape = o["shape"]
    if shape == "cube":
        # The box morph is (2r, 2r, h): r = side/2 and h = side is what makes it
        # a cube, and it is the only combination that does.
        side = float(rng.uniform(*o["side_m"]))
        radius_m, height_m = 0.5 * side, side
    else:
        radius_m = float(rng.uniform(*o["radius_m"]))
        height_m = float(rng.uniform(*o["height_m"]))
    spec = ObjectSpec(
        material=material,
        radius_m=radius_m,
        height_m=height_m,
        mass_kg=float(rng.uniform(*o["mass_kg"])),
        shape=o["shape"],
    )
    if shape == "mesh":
        scale = o["mesh_scale"]
        lo, hi, volume = measure_mesh(
            resolve_mesh_path(str(o["mesh_file"])),
            scale if isinstance(scale, (int, float)) else tuple(float(v) for v in scale),
            o["mesh_euler_deg"],
        )
        spec.aabb_min = (float(lo[0]), float(lo[1]), float(lo[2]))
        spec.aabb_max = (float(hi[0]), float(hi[1]), float(hi[2]))
        spec.volume_override_m3 = float(volume)
        spec.radius_m = 0.5 * float(max(hi[0] - lo[0], hi[1] - lo[1]))
        spec.height_m = float(hi[2] - lo[2])
        spec.base_quat = euler_deg_to_quat_wxyz(o["mesh_euler_deg"])
    return spec


def euler_deg_to_quat_wxyz(euler_deg: Sequence[float]) -> tuple[float, float, float, float]:
    """Extrinsic x-y-z degrees -> wxyz quaternion (scipy's convention).

    Verified against Genesis 1.4.1 the only way that is unambiguous: a bottle whose
    mesh runs z = 0 (base) .. +0.30 (neck) was built with ``quat=Rx(+90)`` and lands
    with its neck at world -y — exactly what scipy's ``Rotation.from_euler('xyz',
    (90, 0, 0))`` predicts, and ``euler=(90, 0, 0)`` gives the same pose to the
    digit. So Genesis' ``euler``, Genesis' ``quat`` and scipy all agree, and the
    scene can convert once and use one convention everywhere.
    """
    from scipy.spatial.transform import Rotation

    q = Rotation.from_euler("xyz", np.asarray(euler_deg, dtype=np.float64), degrees=True).as_quat()
    return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a * b`` (apply b first, then a) for wxyz quaternions."""
    w1, x1, y1, z1 = (float(v) for v in a)
    w2, x2, y2, z2 = (float(v) for v in b)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def resolve_mesh_path(path: str) -> Path:
    """Locate an object mesh: as given, then ``sim/assets/``, then Genesis.

    A configured path is tried verbatim (absolute, or relative to the working
    directory), then relative to the in-repo :data:`ASSETS_DIR` (the default
    home of every mesh the shipped configs use), and only then in the Genesis
    assets directory. The repo-local lookup is what makes a scene portable: the
    configs carry plain relative paths, and the files travel with the code.
    """
    local = Path(path).expanduser()
    if local.is_file():
        return local
    in_repo = ASSETS_DIR / local
    if in_repo.is_file():
        return in_repo
    import genesis as gs

    candidate = Path(gs.utils.get_assets_dir()) / path
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"Object mesh not found: tried '{local}', '{in_repo}' and '{candidate}'. "
        "`objects[].mesh_file` is used as given (absolute or cwd-relative), "
        "then resolved inside sim/assets/, then in the Genesis assets directory."
    )


def measure_mesh(
    path: Path, scale: float | Sequence[float], euler_deg: Sequence[float]
) -> tuple[np.ndarray, np.ndarray, float]:
    """Body-frame AABB (metres) and volume of a mesh body after scale + rotation.

    Mirrors what ``gs.morphs.Mesh(file=..., scale=..., euler=...)`` does to the
    geometry — scale, then the extrinsic x-y-z rotation (scipy convention, the
    one the morph documents) — so the AABB is exact in the frame the morph's
    ``pos`` translates. ``scale`` is a float (uniform) or a 3-tuple (per axis,
    what the morph accepts and what a non-uniform asset such as robosuite's lemon
    needs); the scalar branch is kept verbatim, so every config written before
    tuples existed measures bit-identically. The volume feeds the rigid-body
    density; a mesh that is not watertight (a scan, a mug with an open top) falls
    back to its convex hull, which is what the engine's collision geometry is
    built against anyway.
    """
    import trimesh
    from scipy.spatial.transform import Rotation

    mesh = trimesh.load(str(path), process=False)
    # One file may hold several sub-meshes with their own materials (the wine
    # bottle is glass + label + foil): the engine builds each as its own geom, so
    # the body is their union — AABB over all of them, volume summed.
    parts = list(mesh.geometry.values()) if isinstance(mesh, trimesh.Scene) else [mesh]
    rot = Rotation.from_euler("xyz", np.asarray(euler_deg, dtype=np.float64), degrees=True)
    axes = np.asarray(scale, dtype=np.float64).reshape(-1)
    uniform = axes.size == 1
    factor: float | np.ndarray = float(axes[0]) if uniform else axes
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    volume = 0.0
    for part in parts:
        verts = rot.apply(np.asarray(part.vertices, dtype=np.float64) * factor)
        lo = np.minimum(lo, verts.min(axis=0))
        hi = np.maximum(hi, verts.max(axis=0))
        part_volume = float(part.volume) if bool(part.is_watertight) else float(part.convex_hull.volume)
        volume += abs(part_volume)
    # Scalar keeps the historic expression verbatim (`float(scale) ** 3`) so the
    # measured AABB and volume of every pre-existing config are bit-identical.
    return lo, hi, volume * (float(axes[0]) ** 3 if uniform else float(np.prod(axes)))


def place_object(
    cfg: TaskConfig, spec: ObjectSpec, xy: np.ndarray, yaw: float
) -> tuple[np.ndarray, np.ndarray]:
    """World pose for one object resting in a table slot: ``(pos, wxyz quat)``.

    ``xy`` is the slot *relative to the table centre* (``place_xy`` / a per-episode
    slot + offset), not a world coordinate. The body's AABB bottom face sits on the
    table top and the AABB centre stands over the slot, whatever the authored
    origin is; yaw turns the body about that centre. For a centred primitive (AABB
    centre == origin) this is exactly the historic ``z = table + h/2, xy = slot``
    rule, to the bit.
    """
    from scipy.spatial.transform import Rotation

    lo, hi = spec.local_aabb
    yaw_quat = yaw_to_quat(float(yaw))
    # The body's own orientation (a Y-up mesh asset) rides along with the yaw, so
    # re-placing the object every episode cannot drop it.
    quat = yaw_quat if spec.base_quat is None else _quat_mul_wxyz(yaw_quat, np.asarray(spec.base_quat))
    # scipy takes xyzw; our quats are wxyz. The AABB is already in the base-rotated
    # body frame, so only the yaw moves its centre.
    offset = Rotation.from_quat([yaw_quat[1], yaw_quat[2], yaw_quat[3], yaw_quat[0]]).apply(0.5 * (lo + hi))
    table = cfg["table"]
    pos = np.array(
        [
            float(table["pos"][0]) + float(xy[0]) - offset[0],
            float(table["pos"][1]) + float(xy[1]) - offset[1],
            float(table["height_m"]) - lo[2],
        ],
        dtype=np.float64,
    )
    return pos, quat


def _max_horizontal_reach(o: dict[str, Any]) -> float:
    """Conservative horizontal extent of one object at max sampled size.

    Cylinder: radius. Box: the circumscribed radius ``r*sqrt(2)`` (corners), since
    the placement guard must hold for any yaw. A cube takes the same corner
    factor, over half its own ``side_m`` — ``radius_m``, the box knob, means
    nothing to it. Used only by the separation guard.
    """
    shape = o["shape"]
    r = 0.5 * float(max(o["side_m"])) if shape == "cube" else float(max(o["radius_m"]))
    return r * float(np.sqrt(2.0)) if shape in ("box", "cube") else r


def _min_surface_gap(objs: list[dict[str, Any]], xy_offsets: np.ndarray) -> float:
    """Smallest table-plane surface gap between any two placed slots (inf if <2)."""
    best = float("inf")
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            pi = np.asarray(objs[i]["place_xy"], dtype=np.float64)[:2] + xy_offsets[i]
            pj = np.asarray(objs[j]["place_xy"], dtype=np.float64)[:2] + xy_offsets[j]
            gap = float(np.linalg.norm(pi - pj)) - _max_horizontal_reach(objs[i]) - _max_horizontal_reach(objs[j])
            best = min(best, gap)
    return best


def _tray_clearance_m(cfg: TaskConfig, objs: list[dict[str, Any]]) -> float:
    """Smallest surface gap between a body at a canonical slot and the tray (inf if none).

    The same conservative circle test as :func:`_min_surface_gap` — a body's
    circumradius against the *outer* footprint of the tray — and evaluated over
    every (slot, body) pair, because the slot jitter can put a larger body
    closer to the tray than the slot's own body would be.

    Per-episode jitter is folded in by inflating every slot by the largest
    ``xy_jitter_m`` any body declares: the guard is about the scene a config can
    produce, not about one draw.
    """
    t = cfg["tray"]
    if not t["enabled"] or not objs:
        return float("inf")
    table = cfg["table"]
    cx = float(table["pos"][0]) + float(t["place_xy"][0])
    cy = float(table["pos"][1]) + float(t["place_xy"][1])
    hx = 0.5 * float(t["inner_x_m"]) + float(t["wall_m"])
    hy = 0.5 * float(t["inner_y_m"]) + float(t["wall_m"])
    jitter = max(float(o["xy_jitter_m"]) for o in objs)
    best = float("inf")
    for o in objs:
        sx = float(table["pos"][0]) + float(o["place_xy"][0])
        sy = float(table["pos"][1]) + float(o["place_xy"][1])
        dx = max(abs(sx - cx) - hx - jitter, 0.0)
        dy = max(abs(sy - cy) - hy - jitter, 0.0)
        best = min(best, float(np.hypot(dx, dy)) - _max_horizontal_reach(o))
    return best


def _temperature_profile_sources(
    cfg: TaskConfig, rng: np.random.Generator
) -> tuple[list[int], int]:
    """Body -> temperature-profile source, plus the actual target body.

    Every ``temperature_groups`` entry is a set of interchangeable bodies. One
    permutation per group is drawn here and fixed for the episode, so the
    cold/warm/room profiles swap roles without moving any body. ``source[i]``
    is the object whose ``temp_c`` range body ``i`` is seeded from; the target
    is the body that received the configured ``target_index``'s range. With no
    groups this consumes no draws and returns the configured target.
    """
    n = len(cfg["objects"])
    source = list(range(n))
    for group in cfg["temperature_groups"]:
        perm = rng.permutation(len(group))
        for k, body in enumerate(group):
            source[int(body)] = int(group[int(perm[k])])
    target = next(i for i, src in enumerate(source) if src == int(cfg["target_index"]))
    return source, target


def sample_episode(cfg: TaskConfig, rng: np.random.Generator) -> dict[str, Any]:
    """Per-episode randomization, one row per object (``cfg["objects"]``).

    Returns ``xy_offsets (N,2)``, ``yaws (N,)``, ``temps_c (N,)``, ``slots
    (N,2)``, the actual ``target_index`` and ``temperature_profile_sources
    (N,)``. The target is the body that received the configured
    ``target_index``'s temperature profile; with groups it changes per episode,
    and the language string / ``object_*`` attrs follow it.

    Pose jitter and yaw are drawn per object as before. Temperature profiles are
    permuted within each ``temperature_groups`` entry: bodies exchange their
    ``temp_c`` ranges, so identical bodies swap cold/warm/room roles without
    moving. The group permutation is drawn once, before the pose loop, and is
    fixed across gap retries. An empty group list consumes no permutation draw
    and returns the configured target, so legacy seeds reproduce exactly.

    The per-object draw order is xy jitter (2), yaw (1), temperature (1), the
    same stream every multi-object generation has used.
    """
    objs = cfg["objects"]
    n = len(objs)
    profile_sources, target_index = _temperature_profile_sources(cfg, rng)
    xy_offsets = np.zeros((n, 2), dtype=np.float64)
    yaws = np.zeros(n, dtype=np.float64)
    temps_c = np.zeros(n, dtype=np.float64)
    for attempt in range(5):
        for i, o in enumerate(objs):
            xy_offsets[i] = rng.uniform(-o["xy_jitter_m"], o["xy_jitter_m"], size=2)
            yaws[i] = float(rng.uniform(-o["yaw_jitter_rad"], o["yaw_jitter_rad"]))
            temps_c[i] = float(rng.uniform(*objs[profile_sources[i]]["temp_c"]))
        gaps_ok = _min_surface_gap(objs, xy_offsets) >= cfg["min_object_gap_m"] and _tray_clearance_m(
            cfg, objs
        ) >= cfg["min_object_gap_m"]
        if n < 2 or gaps_ok:
            break
        if attempt == 4:
            # Fallback after 5 clashes: canonical slots, jitter dropped. Better a
            # low-diversity episode than two overlapping bodies.
            xy_offsets[:] = 0.0

    home = np.asarray([o["place_xy"] for o in objs], dtype=np.float64)[:, :2]
    return {
        "xy_offsets": xy_offsets,
        "yaws": yaws,
        "temps_c": temps_c,
        "slots": home.copy(),
        "target_index": int(target_index),
        "temperature_profile_sources": np.asarray(profile_sources, dtype=np.int32),
    }


def yaw_to_quat(yaw: float) -> np.ndarray:
    """Yaw (about world z) -> wxyz quaternion."""
    return np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)], dtype=np.float64)


# cameras


def _renderer_for(cfg: TaskConfig) -> Any:
    """Renderer options for this scene.

    Cameras are opt-in (``cameras.enabled``). The default (no cameras) keeps the
    engine's plain Rasterizer, which is what every no-camera gate/generation run
    has always used. With cameras, the ``RENDERER`` constant in sim/obs.py picks
    the backend (read inside the function, so a script that overrides
    ``sim.obs.RENDERER`` before ``build_scene()`` is honoured):

    * ``raster``   — Rasterizer: GL, works headless, slow but correct. On WSL2 it
                     must run under llvmpipe (``LIBGL_ALWAYS_SOFTWARE=1``); see
                     Gate 3 in docs/architecture.md.
    * ``batch``    — BatchRenderer: GPU, one draw call for all cameras. Loads
                     madrona's Vulkan backend at build time and aborts the
                     process when no Vulkan loader is installed (measured: even
                     ``use_rasterizer=True`` takes that path first), so this is
                     only for a box with a Vulkan loader + ICD.
    * ``raytrace`` — RayTracer: needs OptiX.
    """
    import genesis as gs

    from sim.obs import RENDERER

    if not cfg["cameras"]["enabled"]:
        return gs.renderers.Rasterizer()
    kind = RENDERER.lower()
    if kind in ("batch", "batch_renderer", "madrona"):
        return gs.renderers.BatchRenderer()
    if kind in ("raytrace", "raytracer"):
        return gs.renderers.RayTracer()
    return gs.renderers.Rasterizer()


def build_cameras(scene: Any, hand_link: Any, cfg: TaskConfig) -> dict[str, Any]:
    """Add the configured cameras to a scene that has not been built yet.

    Returns the camera dict that ``SceneBundle.render_cameras()`` renders into
    ``*_rgb`` observations.

    The eye-in-hand camera is *attached* to the hand link with a fixed offset,
    so its pose follows the gripper; the offset is authored in the hand frame
    (``pos``/``lookat``/``up``, same convention as ``scene.add_camera``) and
    converted to a 4x4 here.

    ``attach()`` only records the link and offset — the actual pose can only be
    resolved once the solver exists, so the first pose update happens in
    ``SceneBundle.sync_cameras()`` after ``scene.build()``.
    """
    import genesis as gs

    cams: dict[str, Any] = {}
    c = cfg["cameras"]
    if not c["enabled"]:
        return cams
    res = tuple(int(v) for v in c["resolution"])
    cams["agentview"] = scene.add_camera(
        res=res,
        pos=tuple(c["agentview_pos"]),
        lookat=tuple(c["agentview_lookat"]),
        fov=float(c["fov"]),
        near=float(c["near"]),
        far=float(c["far"]),
        GUI=False,
    )
    if c["eye_in_hand"]:
        offset_T = gs.utils.geom.pos_lookat_up_to_T(
            np.asarray(c["eye_in_hand_pos"], dtype=np.float64),
            np.asarray(c["eye_in_hand_lookat"], dtype=np.float64),
            np.asarray(c["eye_in_hand_up"], dtype=np.float64),
        )
        eye = scene.add_camera(
            res=res,
            pos=tuple(c["eye_in_hand_pos"]),  # pose is replaced by the link attachment
            lookat=tuple(c["eye_in_hand_lookat"]),
            up=tuple(c["eye_in_hand_up"]),
            fov=float(c["eye_in_hand_fov"]),
            near=float(c["near"]),
            far=float(c["far"]),
            GUI=False,
        )
        eye.attach(hand_link, np.asarray(offset_T, dtype=np.float64))
        cams["eye_in_hand"] = eye

    return cams


# ------------------------------------------------------------------- resolution


def resolve_urdf_path(cfg: TaskConfig) -> Path:
    """Locate the robot URDF: as given, then ``sim/assets/``, then Genesis.

    Same lookup order as :func:`resolve_mesh_path`. ``sim/assets/urdf/...`` ships
    the Panda; the Genesis copy is only a fallback for a stripped checkout.
    """
    local = Path(cfg["robot"]["urdf"])
    if local.is_file():
        return local
    in_repo = ASSETS_DIR / local
    if in_repo.is_file():
        return in_repo
    import genesis as gs

    candidate = Path(gs.utils.get_assets_dir()) / cfg["robot"]["urdf"]
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"Robot URDF not found: tried '{local}', '{in_repo}' and '{candidate}'. "
        "If the Genesis assets directory is empty, restore genesis/assets/{urdf,xml,meshes} "
        "from upstream v1.4.1 (see docs/architecture.md §9-R12)."
    )


def _resolve_robot_dofs(robot, cfg: TaskConfig) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Arm + finger dof indices (local) from the config's joint names."""

    def _dof(name: str) -> int:
        return int(robot.get_joint(name).dofs_idx_local[0])

    arm = np.array([_dof(n) for n in cfg["robot"]["arm_joint_names"]], dtype=np.int64)
    fingers = np.array([_dof(n) for n in cfg["robot"]["finger_joint_names"]], dtype=np.int64)
    return arm, fingers, [j.name for j in robot.joints]


def _resolve_links(robot, cfg: TaskConfig) -> tuple[str, tuple[str, str], list[str]]:
    """Hand/finger link names from the config, plus every link name."""
    r = cfg["robot"]
    return r["hand_link_name"], tuple(r["finger_link_names"]), [link.name for link in robot.links]


# --------------------------------------------------------------------- bundle


@dataclass(frozen=True)
class TrayRegion:
    """Where the tray's inner cavity is, in world coordinates (docs §9-R39).

    All numbers come from the build-time geometry (the validated ``tray`` block
    and the table), never from a live entity pose: the parts are fixed, so this
    is exact rather than sampled. The success check for a put-in-tray task reads
    it.
    """

    center_xy: tuple[float, float]
    inner_x_m: float
    inner_y_m: float
    floor_z: float  # z of the cavity floor's top face
    rim_z: float  # z of the walls' top edge

    @property
    def half_x_m(self) -> float:
        return 0.5 * self.inner_x_m

    @property
    def half_y_m(self) -> float:
        return 0.5 * self.inner_y_m

    def contains_xy(self, x: float, y: float) -> bool:
        """Is (x, y) inside the cavity footprint? (Not the walls' thickness.)"""
        return abs(float(x) - self.center_xy[0]) <= self.half_x_m and abs(float(y) - self.center_xy[1]) <= self.half_y_m


@dataclass
class SceneBundle:
    cfg: TaskConfig
    scene: Any
    robot: Any
    objects: list[Any]  # every movable body, in cfg["objects"] order
    layouts: list[FingerLayout]
    arm_dofs: np.ndarray
    finger_dofs: np.ndarray
    hand_link_name: str
    finger_link_names: tuple[str, str]
    tactile: list[TactileHandle]
    link_contacts: list[TactileHandle]
    temperatures: list[TemperatureHandle]
    object_specs: list[ObjectSpec]
    n_envs: int = 1
    # Which object this episode's expert targets; set per episode by
    # reset_episode(). `obj` / `object_spec` below follow it, so every existing
    # single-object call site (expert, success check) is unchanged.
    target_index: int = 0
    joint_names: list[str] = field(default_factory=list)
    link_names: list[str] = field(default_factory=list)
    cameras: dict[str, Any] = field(default_factory=dict)
    # modality -> its own finger layouts (density differs per modality; the
    # `layouts` field above is the primary modality's set).
    tactile_layouts: dict[str, list[FingerLayout]] = field(default_factory=dict)
    # The tray, when the config builds one (docs §9-R39).
    tray: TrayRegion | None = None

    # ------------------------------------------------------------ target object

    @property
    def obj(self) -> Any:
        """The targeted object's entity (follows ``target_index``)."""
        return self.objects[self.target_index]

    @property
    def object_spec(self) -> ObjectSpec:
        """The targeted object's frozen geometry/mass."""
        return self.object_specs[self.target_index]

    # ---------------------------------------------------------------- rendering

    def sync_cameras(self) -> None:
        """Move link-attached cameras to their current link pose (call before rendering)."""
        for cam in self.cameras.values():
            if getattr(cam, "_attached_link", None) is not None:
                cam.move_to_attach()

    def render_cameras(self) -> dict[str, np.ndarray]:
        """Render every camera; returns ``{name: (n_envs, H, W, 3) uint8}``.

        ``scene.render_all_cameras`` only exists for the BatchRenderer, so other
        renderers go through per-camera ``render()`` instead (same frames, one
        draw call per camera). The Rasterizer with ``split_envs=False`` returns
        an unbatched ``(H, W, 3)``, which is normalized to ``(1, H, W, 3)`` so
        callers see one shape whatever the backend.
        """
        if not self.cameras:
            return {}
        self.sync_cameras()
        if self.scene.visualizer.batch_renderer is not None:
            frames = self.scene.render_all_cameras(rgb=True)[0]
        else:
            frames = [
                cam.render(rgb=True, depth=False, segmentation=False, normal=False)[0] for cam in self.cameras.values()
            ]
        out: dict[str, np.ndarray] = {}
        for name, frame in zip(self.cameras, frames, strict=True):
            arr = np.asarray(frame)
            if arr.ndim == 3:
                arr = arr[None]
            out[name] = arr
        return out

    # ------------------------------------------------------------- state access

    def hand_pose(self) -> tuple[np.ndarray, np.ndarray]:
        link = self.robot.get_link(self.hand_link_name)
        pos = np.asarray(link.get_pos().detach().cpu()).reshape(-1)[:3]
        quat = np.asarray(link.get_quat().detach().cpu()).reshape(-1)[:4]
        return pos.astype(np.float64), quat.astype(np.float64)

    @staticmethod
    def _entity_pose(entity: Any) -> tuple[np.ndarray, np.ndarray]:
        pos = np.asarray(entity.get_pos().detach().cpu()).reshape(-1)[:3]
        quat = np.asarray(entity.get_quat().detach().cpu()).reshape(-1)[:4]
        return pos.astype(np.float64), quat.astype(np.float64)

    def object_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """The targeted object's (pos_world, quat_wxyz)."""
        return self._entity_pose(self.obj)

    def object_pose_of(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """One object's (pos_world, quat_wxyz), by index.

        Same float64 read as :meth:`object_pose` -- unlike :meth:`object_poses`,
        which stacks to float32 -- so an expert that walks several candidates
        gets the target's own numbers when it happens to be aiming at the target
        (docs §9-R38).
        """
        return self._entity_pose(self.objects[int(index)])

    def object_poses(self) -> np.ndarray:
        """``(N, 7)`` pos_world + wxyz quat per object, same encoding as link_pose."""
        rows = []
        for entity in self.objects:
            pos, quat = self._entity_pose(entity)
            rows.append(np.concatenate([pos, quat]))
        return np.stack(rows).astype(np.float32)

    def joint_state(self) -> np.ndarray:
        """7 arm + 2 finger joint positions (policy observation)."""
        q = np.asarray(self.robot.get_dofs_position(dofs_idx_local=np.concatenate([self.arm_dofs, self.finger_dofs])).detach().cpu())
        return q.reshape(-1).astype(np.float64)

    # ------------------------------------------------------------------ control

    def step(self) -> None:
        self.scene.step()

    def reset_episode(self, ep: dict[str, Any]) -> None:
        """reset -> robot init pose -> all object poses -> per-object temperature seed.

        ``ep`` is an episode dict from :func:`sample_episode`:
        ``slots (N,2)``, ``xy_offsets (N,2)`` relative to those slots,
        ``yaws (N,)``, ``temps_c (N,)``, ``target_index``.
        """
        self.scene.reset()
        self.set_robot_init_qpos()

        n = len(self.objects)
        self.target_index = int(ep["target_index"])
        xy_offsets = np.asarray(ep["xy_offsets"], dtype=np.float64).reshape(n, 2)
        yaws = np.asarray(ep["yaws"], dtype=np.float64).reshape(n)
        temps_c = np.asarray(ep["temps_c"], dtype=np.float64).reshape(n)
        slot_xy = np.asarray(ep["slots"], dtype=np.float64).reshape(n, 2)

        for i, (entity, spec) in enumerate(zip(self.objects, self.object_specs, strict=True)):
            slot = np.array(
                [float(slot_xy[i, 0]) + float(xy_offsets[i, 0]), float(slot_xy[i, 1]) + float(xy_offsets[i, 1])],
                dtype=np.float64,
            )
            pos, quat = place_object(self.cfg, spec, slot, float(yaws[i]))
            entity.set_pos(pos)
            entity.set_quat(quat)

        for handle in self.temperatures:
            if handle.kind == "object" and handle.object_index is not None:
                seed_temperature(handle.sensor, float(temps_c[handle.object_index]))

    def set_robot_init_qpos(self) -> None:
        q = np.asarray(self.cfg["robot"]["init_qpos"], dtype=np.float64)
        self.robot.set_dofs_position(q, zero_velocity=True)
        self.robot.control_dofs_position(q)


# build scene

def build_scene(
    cfg: TaskConfig,
    seed: int = 0,
    n_envs: int = 1,
    show_viewer: bool = False,
    logging_level: str = "warning",
) -> SceneBundle:
    import genesis as gs

    if not gs._initialized:
        gs.init(backend=gs.gpu, precision="32", seed=seed, logging_level=logging_level)
    
    # 传感器补丁
    install_sensor_reset_fix()
    # 贴图补丁
    install_texture_resolver_policy()

    rng = np.random.default_rng(seed)
    object_cfgs = cfg["objects"]
    materials = list(cfg["material_list"])
    # One spec per object; drawn in cfg["objects"] order, with the config's
    # per-object materials (validated on load).
    specs = [
        sample_object_spec(o, m, rng)
        for m, o in zip(materials, object_cfgs, strict=True)
    ]

    # options
    rigid_options = gs.options.RigidOptions(
        box_box_detection=True,
        enable_self_collision=False,
        enable_joint_limit=True,
        friction_cone=gs.friction_cone.elliptic,
        contact_resolution=gs.contact_resolution.signorini,
        constraint_timeconst=0.01,
        iterations=50,
        ls_iterations=50,
        use_gjk_collision=True,
        enable_multi_contact=True,
        max_collision_pairs=64,
        multiplier_collision_broad_phase=16,
        batch_links_info=True,
    )
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=cfg["controller"]["sim_dt"], substeps=1),
        rigid_options=rigid_options,
        show_viewer=show_viewer,
        renderer=_renderer_for(cfg),
    )

    # ---------------------------------------------------------------- entities
    table_cfg = cfg["table"]
    table_color = table_cfg.get("color")
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(
        gs.morphs.Box(
            size=tuple(table_cfg["size"]),
            pos=(table_cfg["pos"][0], table_cfg["pos"][1], table_cfg["height_m"] - 0.5 * table_cfg["size"][2]),
            fixed=True,
        ),
        # Render-only: the surface carries no physics/thermal parameters (those
        # live in `material` / the temperature config), so recoloring the table
        # changes nothing but the RGB frames.
        surface=gs.surfaces.Default(color=tuple(table_color)) if table_color is not None else None,
    )
    robot_cfg = cfg["robot"]
    robot = scene.add_entity(
        gs.morphs.URDF(
            # Resolved to an absolute path (sim/assets/urdf/... first): Genesis
            # joins a *relative* URDF path with its own assets dir, so the
            # repo-local copy must be handed over absolute (docs §6).
            file=str(resolve_urdf_path(cfg)),
            pos=tuple(robot_cfg["base_pos"]),
            quat=tuple(robot_cfg["base_quat"]),
            fixed=True,
            merge_fixed_links=robot_cfg["merge_fixed_links"],
        )
    )
    colors = [o.get("color") for o in object_cfgs]

    objects = []
    for o_cfg, spec, color in zip(object_cfgs, specs, colors, strict=True):
        # Built at the canonical slot (place_xy), NOT the per-episode pose: the
        # temperature sensors freeze the link AABB at build time, so an object may
        # only move by its per-episode jitter afterwards (docs §9-R21).
        pos, _ = place_object(cfg, spec, np.asarray(o_cfg["place_xy"], dtype=np.float64)[:2], 0.0)
        pos = (float(pos[0]), float(pos[1]), float(pos[2]))
        shape = spec.shape
        if shape in ("box", "cube"):
            # A cube reaches this same expression: `sample_object_spec` gave it
            # r = side/2 and h = side, so (2r, 2r, h) *is* the side-long cube.
            morph = gs.morphs.Box(size=(2.0 * spec.radius_m, 2.0 * spec.radius_m, spec.height_m), pos=pos)
        elif shape == "cylinder":
            morph = gs.morphs.Cylinder(radius=spec.radius_m, height=spec.height_m, pos=pos)
        else:
            # Real object mesh (a LIBERO/robosuite asset, say): the engine's mesh
            # parser reads the OBJ's material and texture image, so the RGB frames
            # show the asset as it is. `align=False` keeps the authored frame --
            # `pos` above is what places the AABB -- and the collision geometry is
            # a coacd decomposition of the decimated mesh (non-convex bodies such
            # as a bowl or a mug handle need more than one hull).
            morph = gs.morphs.Mesh(
                file=str(resolve_mesh_path(str(o_cfg["mesh_file"]))),
                # float or 3-tuple, the same two forms `measure_mesh` above took
                # (the AABB and the geometry must scale the same way).
                scale=(
                    float(o_cfg["mesh_scale"])
                    if isinstance(o_cfg["mesh_scale"], (int, float))
                    else tuple(float(v) for v in o_cfg["mesh_scale"])
                ),
                # The alias-corrected orientation, same convention as the quat
                # `place_object` re-issues every episode (verified: euler and quat
                # agree with scipy, sim/scene.py::euler_deg_to_quat_wxyz).
                quat=tuple(spec.base_quat) if spec.base_quat is not None else None,
                pos=pos,
                align=False,
                convexify=True,
            )
        objects.append(
            scene.add_entity(
                morph,
                material=gs.materials.Rigid(rho=spec.mass_kg / spec.volume_m3, friction=1.0),
                # Render-only surface color, same convention as the table below:
                # physics and thermal properties live in `material` / the temperature
                # config, so this changes nothing but the RGB frames. None keeps the
                # engine default — the material-judging twins must stay
                # indistinguishable by appearance (docs §9-R38).
                surface=gs.surfaces.Default(color=tuple(color)) if color is not None else None,
            )
        )

    # tray
    # Built *after* the objects on purpose: entity indices are referenced by the
    # sensors and the recorded trajectories, so a new entity may only ever be
    # appended. Five `fixed=True` boxes — Genesis has no compound primitive morph
    # (docs §9-R39) — a floor flush with the table top and four walls rising
    # `rim_m` above it. Static: no dofs, no sensors; the expert lowers the body
    # until it rests on the floor, and the walls are what stop it leaving.
    tray_region: TrayRegion | None = None
    t = cfg["tray"]
    if t["enabled"]:
        cx = float(table_cfg["pos"][0]) + float(t["place_xy"][0])
        cy = float(table_cfg["pos"][1]) + float(t["place_xy"][1])
        top = float(table_cfg["height_m"])
        hx, hy = 0.5 * float(t["inner_x_m"]), 0.5 * float(t["inner_y_m"])
        wall, rim, floor = float(t["wall_m"]), float(t["rim_m"]), float(t["floor_m"])
        # Parts are sunk `fuse_m` into their neighbours so no two faces are
        # coplanar (coincident faces are free to z-fight in the rasterizer).
        sink = float(t["fuse_m"])
        floor_z = tray_floor_z(t, top)  # the surface the placed body rests on
        floor_cz = floor_z - 0.5 * floor
        wall_z0, wall_h = floor_z - sink, rim + sink
        wall_cz = wall_z0 + 0.5 * wall_h
        surface = gs.surfaces.Default(color=tuple(t["color"]))
        parts = [
            # floor plate, its edges tucked inside the walls
            ((t["inner_x_m"] + 2 * wall - sink, t["inner_y_m"] + 2 * wall - sink, floor), (cx, cy, floor_cz)),
            # the two x-walls run the full outer y; the y-walls tuck into them
            ((wall, t["inner_y_m"] + 2 * wall, wall_h), (cx - hx - 0.5 * wall, cy, wall_cz)),
            ((wall, t["inner_y_m"] + 2 * wall, wall_h), (cx + hx + 0.5 * wall, cy, wall_cz)),
            ((t["inner_x_m"] + sink, wall, wall_h), (cx, cy - hy - 0.5 * wall, wall_cz)),
            ((t["inner_x_m"] + sink, wall, wall_h), (cx, cy + hy + 0.5 * wall, wall_cz)),
        ]
        for size, box_pos in parts:
            scene.add_entity(gs.morphs.Box(size=tuple(size), pos=tuple(box_pos), fixed=True), surface=surface)
        tray_region = TrayRegion(
            center_xy=(cx, cy),
            inner_x_m=float(t["inner_x_m"]),
            inner_y_m=float(t["inner_y_m"]),
            floor_z=floor_z,
            rim_z=tray_rim_z(t, top),
        )

    # sensors
    urdf_path = resolve_urdf_path(cfg)
    hand_link_name, finger_link_names, link_names = _resolve_links(robot, cfg)
    modalities = tuple(cfg["tactile"]["modalities"])
    # One layout set per *distinct* probe grid: pad geometry (face, extents,
    # normal) is grid-independent, only the sampled positions differ, so the
    # dense 16x16 marker grid and the coarse 4x3 force grid are separate
    # derivations over the same pad (docs §5.2). The force layout's pad normal
    # is also what decomposes the summed probes into the 6D resultant.
    grids = list(dict.fromkeys(grid_for(m) for m in modalities))
    layouts_by_grid = {
        g: derive_finger_layout(
            urdf_path,
            tuple(finger_link_names),
            probe_grid=tuple(g),
            probe_offset_m=PROBE_OFFSET_M,
            pad_tol_m=PAD_TOL_M,
        )
        for g in grids
    }
    layouts_by_modality = {m: layouts_by_grid[grid_for(m)] for m in modalities}
    # `bundle.layouts` stays the primary modality's grid: the temperature
    # grids use it as "one entry per finger".
    layouts = layouts_by_modality[primary_modality(modalities)]
    # Every object is an indenter for the point-cloud modalities (`elastomer`,
    # `proximity`): their SDF queries run against each tracked link's collision
    # geometry. `_build_options` only forwards this for those two modalities, so
    # it is inert for force (docs §5.2).
    track_link_idx = tuple(int(o.links[0].idx) for o in objects)
    # n_sample_points is a *per-link* budget: an int is split across tracked
    # links (two objects would halve the marker density), so pass one count per
    # link explicitly.
    per_link_points = [int(PROXIMITY_SAMPLE_POINTS)] * len(objects)
    elastomer_params = {"n_sample_points": list(per_link_points)}
    elastomer_params.update(ELASTOMER_PARAMS)
    tactile = build_finger_sensors(
        scene,
        entity=robot,
        layouts_by_modality=layouts_by_modality,
        modalities=modalities,
        track_link_idx=track_link_idx,
        decimation=cfg["controller"]["decimation"],
        # Sampling density of that point cloud sets the proximity ramp's
        # granularity (docs §9-R22 follow-up).
        extra_params={
            "proximity": {"n_sample_points": list(per_link_points)},
            # Gel-sensor shading parameters come straight from the module
            # constants (docs §9-R8).
            "elastomer": elastomer_params,
        },
    )
    link_contacts = (
        build_link_contact_sensors(scene, entity=robot, link_names=list(finger_link_names))
        if CONTACT_LINKS
        else []
    )

    # Temperature: one shared properties_dict for every sensor in the scene.
    object_links = [o.links[0] for o in objects]
    finger_links = [robot.get_link(n) for n in finger_link_names]
    temp_cfg = cfg["temperature"]
    # All sensors carry base temperature = ambient; each object's grid is
    # re-seeded per episode on top of it. make_properties_dict() takes dicts or
    # ready-made TemperatureProperties and passes the latter through unchanged.
    # Each object link carries its own body's material.
    ambient = float(temp_cfg["ambient_temp_c"])
    properties_dict = make_properties_dict(
        {
            **{
                int(link.idx): properties_from_material(dict(FINGER_PROPERTIES), base_temperature_c=ambient)
                for link in finger_links
            },
            **{
                int(link.idx): properties_from_material(
                    dict(cfg["material_properties"][materials[i]]), base_temperature_c=ambient
                )
                for i, link in enumerate(object_links)
            },
        },
        default=properties_from_material(dict(ENVIRONMENT_PROPERTIES), base_temperature_c=ambient),
    )
    finger_grid = tuple(int(v) for v in temp_cfg["finger_grid"])
    object_grid = tuple(int(v) for v in temp_cfg["object_grid"])

    temperatures = build_finger_temperature_sensors(
        scene,
        entity=robot,
        finger_grid=finger_grid,
        ambient_temp_c=ambient,
        finger_link_names=list(finger_link_names),
        properties_dict=properties_dict,
    )
    for i, entity in enumerate(objects):
        temperatures.append(
            build_object_temperature_sensor(
                scene,
                entity=entity,
                object_grid=object_grid,
                ambient_temp_c=ambient,
                properties_dict=properties_dict,
                link_name=f"object_{i}",
                # `temperature_object_<i>` is an in-process handle key; the
                # recorded channels under obs/temperature/ keep their names.
                name=f"temperature_object_{i}",
                object_index=i,
            )
        )

    # Contact-area policy (docs §9-R37): the engine's own per-contact areas are
    # replaced for finger contacts by the pad-face split A_i = A_face/sqrt(n_i),
    # so both fingers of a grasp carry the same tip temperature in every contact
    # phase (engine: the 2- vs 3-point phases differ 24x). The policy is fixed
    # in sim/sensors/temperature.py, so every scene installs the same split.
    finger_faces = {
        int(link.idx): lay.pad_face_area_m2
        for link, lay in zip(finger_links, layouts, strict=True)
    }
    install_contact_area_policy(finger_faces)
    faces = " ".join(f"{name} {a * 1e6:.3f}" for name, a in zip(finger_link_names, finger_faces.values()))
    print(f"  contact area policy pad_face: {faces} mm^2 -> A_i = A/sqrt(n_i), sum(A_i^2) = A^2 (docs §9-R37)")

    # ------------------------------------------------------------------ cameras
    # Cameras must be declared before build; the eye-in-hand pose can only be
    # resolved afterwards (it needs the built solver's link poses).
    cameras = build_cameras(scene, robot.get_link(hand_link_name), cfg)

    # ------------------------------------------------------------------- build
    scene.build(n_envs=n_envs)

    arm_dofs, finger_dofs, joint_names = _resolve_robot_dofs(robot, cfg)
    robot.set_dofs_kp(np.asarray(robot_cfg["arm_kp"], dtype=np.float32), arm_dofs)
    robot.set_dofs_kv(np.asarray(robot_cfg["arm_kv"], dtype=np.float32), arm_dofs)
    robot.set_dofs_kp(np.asarray(robot_cfg["finger_kp"], dtype=np.float32), finger_dofs)
    robot.set_dofs_kv(np.asarray(robot_cfg["finger_kv"], dtype=np.float32), finger_dofs)
    robot.set_dofs_force_range(
        np.asarray(robot_cfg["arm_force_min"], dtype=np.float32),
        np.asarray(robot_cfg["arm_force_max"], dtype=np.float32),
        arm_dofs,
    )
    robot.set_dofs_force_range(
        np.asarray(robot_cfg["finger_force_min"], dtype=np.float32),
        np.asarray(robot_cfg["finger_force_max"], dtype=np.float32),
        finger_dofs,
    )

    bundle = SceneBundle(
        cfg=cfg,
        scene=scene,
        robot=robot,
        objects=objects,
        layouts=layouts,
        tactile_layouts=layouts_by_modality,
        arm_dofs=arm_dofs,
        finger_dofs=finger_dofs,
        hand_link_name=hand_link_name,
        finger_link_names=finger_link_names,
        tactile=tactile,
        link_contacts=link_contacts,
        temperatures=temperatures,
        object_specs=specs,
        n_envs=n_envs,
        joint_names=joint_names,
        link_names=link_names,
        cameras=cameras,
        tray=tray_region,
    )
    bundle.set_robot_init_qpos()
    bundle.sync_cameras()
    return bundle

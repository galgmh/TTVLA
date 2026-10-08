"""Scene construction: Franka Panda + parallel gripper, table, hot object, sensors.

Contract highlights (verified against Genesis 1.4.1):

* The robot is built at its authored pose; the object is built at a canonical
  pose and repositioned per episode through its root DOF, because temperature
  sensors freeze the link AABB / voxel volume at build time.
* All TemperatureGrid sensors in a scene receive identical shared-metadata
  values (ambient / convection / simulate_all_link_temperatures) — the engine
  keeps "last non-None wins" per scene.
* Per-episode object temperature goes through seed_temperature() AFTER
  scene.reset() and BEFORE the first scene.step() of the episode.
* Object size/mass and material are sampled once per scene build (geometry is
  fixed at build); pose/temperature are sampled per episode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tempvla.compat import install_sensor_reset_fix
from tempvla.config import ObjectConfig, TaskConfig
from tempvla.sensors import (
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


# --------------------------------------------------------------------- sampling


@dataclass
class ObjectSpec:
    """One object's frozen geometry/mass (sampled once per scene build).

    ``radius_m`` is the cylinder radius or the box *half-width*; the box morph is
    ``(2r, 2r, h)``. ``volume_m3`` feeds the rigid-body density
    (``mass_kg / volume``), so the shape must be accounted for there.
    """

    material: str
    radius_m: float
    height_m: float
    mass_kg: float
    shape: str = "cylinder"

    @property
    def volume_m3(self) -> float:
        if self.shape == "box":
            return float((2.0 * self.radius_m) ** 2 * self.height_m)
        return float(np.pi * self.radius_m**2 * self.height_m)


def sample_object_spec(
    cfg: TaskConfig, material: str, rng: np.random.Generator, *, object_cfg: ObjectConfig | None = None
) -> ObjectSpec:
    """Sample one object's frozen geometry/mass from its ObjectConfig (or cfg.object)."""
    o = object_cfg if object_cfg is not None else cfg.object
    return ObjectSpec(
        material=material,
        radius_m=float(rng.uniform(*o.radius_m)),
        height_m=float(rng.uniform(*o.height_m)),
        mass_kg=float(rng.uniform(*o.mass_kg)),
        shape=str(o.shape),
    )


def _max_horizontal_reach(o: ObjectConfig) -> float:
    """Conservative horizontal extent of one object at max sampled size.

    Cylinder: radius. Box: the circumscribed radius ``r*sqrt(2)`` (corners), since
    the placement guard must hold for any yaw. Used only by the separation guard.
    """
    r = float(max(o.radius_m))
    return r * float(np.sqrt(2.0)) if str(o.shape).lower() == "box" else r


def _min_surface_gap(objs: list[ObjectConfig], xy_offsets: np.ndarray) -> float:
    """Smallest table-plane surface gap between any two placed slots (inf if <2)."""
    best = float("inf")
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            pi = np.asarray(objs[i].place_xy, dtype=np.float64)[:2] + xy_offsets[i]
            pj = np.asarray(objs[j].place_xy, dtype=np.float64)[:2] + xy_offsets[j]
            gap = float(np.linalg.norm(pi - pj)) - _max_horizontal_reach(objs[i]) - _max_horizontal_reach(objs[j])
            best = min(best, gap)
    return best


def sample_episode(cfg: TaskConfig, rng: np.random.Generator) -> dict[str, Any]:
    """Per-episode randomization, one row per object (``cfg.object_list()``).

    Returns ``xy_offsets (N,2)``, ``yaws (N,)``, ``temps_c (N,)`` and the
    ``target_index`` the scripted expert will pick — the language string and the
    legacy ``object_*`` attrs follow the target. For N=1 the draw order and the
    final rng state are exactly the pre-multi-object ones (the target draw is
    skipped, not drawn-and-discarded), so existing seeds reproduce their
    episodes bit for bit, expert jitter included.
    """
    objs = cfg.object_list()
    n = len(objs)
    xy_offsets = np.zeros((n, 2), dtype=np.float64)
    yaws = np.zeros(n, dtype=np.float64)
    temps_c = np.zeros(n, dtype=np.float64)
    for attempt in range(5):
        for i, o in enumerate(objs):
            xy_offsets[i] = rng.uniform(-o.xy_jitter_m, o.xy_jitter_m, size=2)
            yaws[i] = float(rng.uniform(-o.yaw_jitter_rad, o.yaw_jitter_rad))
            temps_c[i] = float(rng.uniform(*o.temp_c))
        if n < 2 or _min_surface_gap(objs, xy_offsets) >= cfg.min_object_gap_m:
            break
        if attempt == 4:
            # Fallback after 5 clashes: canonical slots, jitter dropped. Better a
            # low-diversity episode than two overlapping bodies.
            xy_offsets[:] = 0.0
    return {
        "xy_offsets": xy_offsets,
        "yaws": yaws,
        "temps_c": temps_c,
        # Skipped for a single object so the rng stream (and hence the expert's
        # own jitter draw) is untouched relative to pre-multi-object runs.
        "target_index": 0 if n == 1 else int(rng.integers(0, n)),
    }


def scripted_episode(
    cfg: TaskConfig,
    *,
    xy_offset: Any = None,
    yaw: float = 0.0,
    temp_c: float = 0.0,
    target_index: int = 0,
) -> dict[str, Any]:
    """One fully-specified episode dict for gates/probes (no rng).

    The target object is placed at ``xy_offset``/``yaw``/``temp_c`` relative to the
    table center; every other object stays at its canonical ``place_xy`` slot,
    yaw 0, ambient temperature. With a single-object config this is exactly the
    old ``reset_episode(xy_offset=..., yaw=..., temp_c=...)`` call.
    """
    objs = cfg.object_list()
    n = len(objs)
    target = int(target_index) % n
    xy_offsets = np.zeros((n, 2), dtype=np.float64)
    yaws = np.zeros(n, dtype=np.float64)
    temps_c = np.full(n, float(cfg.temperature.ambient_temp_c), dtype=np.float64)
    if xy_offset is not None:
        xy_offsets[target] = np.asarray(xy_offset, dtype=np.float64).reshape(-1)[:2]
    yaws[target] = float(yaw)
    temps_c[target] = float(temp_c)
    return {"xy_offsets": xy_offsets, "yaws": yaws, "temps_c": temps_c, "target_index": target}


def yaw_to_quat(yaw: float) -> np.ndarray:
    """Yaw (about world z) -> wxyz quaternion."""
    return np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)], dtype=np.float64)


# --------------------------------------------------------------------- cameras


def _renderer_for(cfg: TaskConfig) -> Any:
    """Renderer options for this scene.

    Cameras are opt-in (``cameras.enabled``). The default (no cameras) keeps the
    engine's plain Rasterizer, which is what every no-camera gate/generation run
    has always used. With cameras, ``cameras.renderer`` picks the backend:

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

    if not cfg.cameras.enabled:
        return gs.renderers.Rasterizer()
    kind = str(cfg.cameras.renderer).lower()
    if kind in ("batch", "batch_renderer", "madrona"):
        return gs.renderers.BatchRenderer()
    if kind in ("raytrace", "raytracer"):
        return gs.renderers.RayTracer()
    if kind not in ("raster", "rasterizer"):
        raise ValueError(f"unknown cameras.renderer {cfg.cameras.renderer!r}")
    return gs.renderers.Rasterizer()


def build_cameras(scene: Any, hand_link: Any, cfg: TaskConfig) -> dict[str, Any]:
    """Add the configured cameras to a scene that has not been built yet.

    The eye-in-hand camera is *attached* to the hand link with a fixed offset,
    so its pose follows the gripper; the offset is authored in the hand frame
    (``pos``/``lookat``/``up``, same convention as ``scene.add_camera``) and
    converted to a 4x4 here.

    ``attach()`` only records the link and offset — the actual pose can only be
    resolved once the solver exists, so the first pose update happens in
    ``SceneBundle.sync_cameras()`` after ``scene.build()``.
    """
    import genesis as gs

    if not cfg.cameras.enabled:
        return {}
    c = cfg.cameras
    res = tuple(int(v) for v in c.resolution)
    cams: dict[str, Any] = {
        "agentview": scene.add_camera(
            res=res,
            pos=tuple(c.agentview_pos),
            lookat=tuple(c.agentview_lookat),
            fov=float(c.fov),
            near=float(c.near),
            far=float(c.far),
            GUI=False,
        )
    }
    if not c.eye_in_hand:
        return cams
    offset_T = gs.utils.geom.pos_lookat_up_to_T(
        np.asarray(c.eye_in_hand_pos, dtype=np.float64),
        np.asarray(c.eye_in_hand_lookat, dtype=np.float64),
        np.asarray(c.eye_in_hand_up, dtype=np.float64),
    )
    eye = scene.add_camera(
        res=res,
        pos=tuple(c.eye_in_hand_pos),  # pose is replaced by the link attachment
        lookat=tuple(c.eye_in_hand_lookat),
        up=tuple(c.eye_in_hand_up),
        fov=float(c.eye_in_hand_fov),
        near=float(c.near),
        far=float(c.far),
        GUI=False,
    )
    eye.attach(hand_link, np.asarray(offset_T, dtype=np.float64))
    cams["eye_in_hand"] = eye
    return cams


# ------------------------------------------------------------------- resolution


def resolve_urdf_path(cfg: TaskConfig) -> Path:
    """Locate the robot URDF on the local FS (project-local first, then assets)."""
    local = Path(cfg.robot.urdf)
    if local.is_file():
        return local
    import genesis as gs

    candidate = Path(gs.utils.get_assets_dir()) / cfg.robot.urdf
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"Robot URDF not found: tried '{local}' and '{candidate}'. "
        "If the Genesis assets directory is empty, restore genesis/assets/{urdf,xml,meshes} "
        "from upstream v1.4.1 (see docs/architecture.md §9-R12)."
    )


def _resolve_robot_dofs(robot, cfg: TaskConfig) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Arm + finger dof indices (local), config names first, heuristic fallback."""
    joint_names = [j.name for j in robot.joints]

    def _dof(name: str) -> int:
        return int(robot.get_joint(name).dofs_idx_local[0])

    try:
        arm = np.array([_dof(n) for n in cfg.robot.arm_joint_names], dtype=np.int64)
        fingers = np.array([_dof(n) for n in cfg.robot.finger_joint_names], dtype=np.int64)
        return arm, fingers, joint_names
    except Exception:  # noqa: BLE001 - fall through to auto-detection
        pass

    actuated = [j for j in robot.joints if getattr(j, "n_dofs", 0) > 0]
    finger_j = [j for j in actuated if "finger" in j.name.lower()]
    arm_j = [j for j in actuated if j not in finger_j]
    if len(arm_j) != 7 or len(finger_j) != 2:
        raise RuntimeError(
            f"Auto joint detection failed (arm={[j.name for j in arm_j]}, "
            f"fingers={[j.name for j in finger_j]}). All joints: {joint_names}"
        )
    return (
        np.array([int(j.dofs_idx_local[0]) for j in arm_j], dtype=np.int64),
        np.array([int(j.dofs_idx_local[0]) for j in finger_j], dtype=np.int64),
        joint_names,
    )


def _resolve_links(robot, cfg: TaskConfig) -> tuple[str, tuple[str, str], list[str]]:
    names = [link.name for link in robot.links]
    if cfg.robot.hand_link_name in names and all(n in names for n in cfg.robot.finger_link_names):
        return cfg.robot.hand_link_name, tuple(cfg.robot.finger_link_names), names
    fingers = [n for n in names if "finger" in n.lower()]
    hand = [n for n in names if "hand" in n.lower()]
    if len(fingers) == 2 and len(hand) == 1:
        return hand[0], (fingers[0], fingers[1]), names
    raise RuntimeError(f"Could not resolve hand/finger links. Links: {names}")


# --------------------------------------------------------------------- bundle


@dataclass
class SceneBundle:
    cfg: TaskConfig
    scene: Any
    robot: Any
    objects: list[Any]  # every movable body, in cfg.object_list() order
    table: Any
    plane: Any
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
    # single-object call site (expert, success check, probes) is unchanged.
    target_index: int = 0
    joint_names: list[str] = field(default_factory=list)
    link_names: list[str] = field(default_factory=list)
    episode: dict[str, Any] = field(default_factory=dict)
    layout_info: dict[str, Any] = field(default_factory=dict)
    cameras: dict[str, Any] = field(default_factory=dict)
    # modality -> its own finger layouts (density differs per modality; the
    # `layouts` field above is the primary modality's set).
    tactile_layouts: dict[str, list[FingerLayout]] = field(default_factory=dict)

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

    def object_poses(self) -> np.ndarray:
        """``(N, 7)`` pos_world + wxyz quat per object, same encoding as link_pose."""
        rows = []
        for entity in self.objects:
            pos, quat = self._entity_pose(entity)
            rows.append(np.concatenate([pos, quat]))
        return np.stack(rows).astype(np.float32)

    def get_qpos(self) -> np.ndarray:
        """Full robot qpos, flat (n_dofs,)."""
        q = np.asarray(self.robot.get_qpos().detach().cpu())
        return q.reshape(-1).astype(np.float64)

    def arm_qpos(self) -> np.ndarray:
        q = np.asarray(self.robot.get_dofs_position(dofs_idx_local=self.arm_dofs).detach().cpu())
        return q.reshape(-1).astype(np.float64)

    def joint_state(self) -> np.ndarray:
        """7 arm + 2 finger joint positions (policy observation)."""
        q = np.asarray(self.robot.get_dofs_position(dofs_idx_local=np.concatenate([self.arm_dofs, self.finger_dofs])).detach().cpu())
        return q.reshape(-1).astype(np.float64)

    # ------------------------------------------------------------------ control

    def step(self) -> None:
        self.scene.step()

    def reset_episode(self, ep: dict[str, Any]) -> None:
        """reset -> robot init pose -> all object poses -> per-object temperature seed.

        ``ep`` is an episode dict from :func:`sample_episode` /
        :func:`scripted_episode`: ``xy_offsets (N,2)`` relative to each object's
        ``place_xy`` slot, ``yaws (N,)``, ``temps_c (N,)``, ``target_index``.
        """
        self.scene.reset()
        self.set_robot_init_qpos()

        n = len(self.objects)
        self.target_index = int(ep.get("target_index", 0)) % n
        xy_offsets = np.asarray(ep["xy_offsets"], dtype=np.float64).reshape(n, 2)
        yaws = np.asarray(ep["yaws"], dtype=np.float64).reshape(n)
        temps_c = np.asarray(ep["temps_c"], dtype=np.float64).reshape(n)

        placed = []
        for i, (entity, o_cfg, spec) in enumerate(
            zip(self.objects, self.cfg.object_list(), self.object_specs, strict=True)
        ):
            x = self.cfg.table.pos[0] + float(o_cfg.place_xy[0]) + float(xy_offsets[i, 0])
            y = self.cfg.table.pos[1] + float(o_cfg.place_xy[1]) + float(xy_offsets[i, 1])
            z = self.cfg.table.height_m + 0.5 * spec.height_m
            entity.set_pos(np.array([x, y, z]))
            entity.set_quat(yaw_to_quat(float(yaws[i])))
            placed.append((x, y))

        self.episode = {
            "xy": placed,  # one (x, y) per object; also the legacy single-object tuple
            "yaw": yaws.copy(),
            "temp_c": temps_c.copy(),
            "target_index": self.target_index,
        }
        for handle in self.temperatures:
            if handle.kind == "object" and handle.object_index is not None:
                seed_temperature(handle.sensor, float(temps_c[handle.object_index]))

    def set_robot_init_qpos(self) -> None:
        q = np.asarray(self.cfg.robot.init_qpos, dtype=np.float64)
        self.robot.set_dofs_position(q, zero_velocity=True)
        self.robot.control_dofs_position(q)

    def object_center_height(self) -> float:
        return float(self.cfg.table.height_m + 0.5 * self.object_spec.height_m)


# ---------------------------------------------------------------- build scene


def build_scene(
    cfg: TaskConfig,
    *,
    material: str | None = None,
    seed: int = 0,
    n_envs: int = 1,
    show_viewer: bool = False,
    logging_level: str = "warning",
) -> SceneBundle:
    import genesis as gs

    if not gs._initialized:
        gs.init(backend=gs.gpu, precision="32", seed=seed, logging_level=logging_level)
    install_sensor_reset_fix()

    rng = np.random.default_rng(seed)
    material = material or (cfg.materials[0] if cfg.materials else "aluminum")
    if material not in cfg.material_properties:
        raise KeyError(f"Material '{material}' missing from material_properties")
    # One spec per object; drawn in object_list() order, so a single-object
    # config samples exactly the values the pre-multi-object code sampled.
    object_cfgs = cfg.object_list()
    specs = [sample_object_spec(cfg, material, rng, object_cfg=o) for o in object_cfgs]

    # ---------------------------------------------------------------- options
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
        sim_options=gs.options.SimOptions(dt=cfg.controller.sim_dt, substeps=1),
        rigid_options=rigid_options,
        show_viewer=show_viewer,
        renderer=_renderer_for(cfg),
    )

    # ---------------------------------------------------------------- entities
    plane = scene.add_entity(gs.morphs.Plane())
    table = scene.add_entity(
        gs.morphs.Box(
            size=tuple(cfg.table.size),
            pos=(cfg.table.pos[0], cfg.table.pos[1], cfg.table.height_m - 0.5 * cfg.table.size[2]),
            fixed=True,
        ),
        # Render-only: the surface carries no physics/thermal parameters (those
        # live in `material` / the temperature config), so recoloring the table
        # changes nothing but the RGB frames.
        surface=gs.surfaces.Default(color=tuple(cfg.table.color)) if cfg.table.color is not None else None,
    )
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=cfg.robot.urdf,
            pos=tuple(cfg.robot.base_pos),
            quat=tuple(cfg.robot.base_quat),
            fixed=True,
            merge_fixed_links=cfg.robot.merge_fixed_links,
        )
    )
    objects = []
    for o_cfg, spec in zip(object_cfgs, specs, strict=True):
        # Built at the canonical slot (place_xy), NOT the per-episode pose: the
        # temperature sensors freeze the link AABB at build time, so an object may
        # only move by its per-episode jitter afterwards (docs §9-R21).
        pos = (
            cfg.table.pos[0] + float(o_cfg.place_xy[0]),
            cfg.table.pos[1] + float(o_cfg.place_xy[1]),
            cfg.table.height_m + 0.5 * spec.height_m,
        )
        shape = str(spec.shape).lower()
        if shape == "box":
            morph = gs.morphs.Box(size=(2.0 * spec.radius_m, 2.0 * spec.radius_m, spec.height_m), pos=pos)
        elif shape == "cylinder":
            morph = gs.morphs.Cylinder(radius=spec.radius_m, height=spec.height_m, pos=pos)
        else:
            raise ValueError(f"Unsupported object shape {spec.shape!r} (cylinder | box)")
        objects.append(
            scene.add_entity(morph, material=gs.materials.Rigid(rho=spec.mass_kg / spec.volume_m3, friction=1.0))
        )

    # ---------------------------------------------------------------- sensors
    urdf_path = resolve_urdf_path(cfg)
    hand_link_name, finger_link_names, link_names = _resolve_links(robot, cfg)
    layout_info: dict[str, Any] = {}
    # One layout set per *distinct* probe grid: pad geometry (face, extents,
    # normal) is grid-independent, only the sampled positions differ, so the
    # dense gel marker grid and the coarse force/depth grid are separate
    # derivations over the same pad (docs §5.2).
    grids = list(dict.fromkeys(cfg.tactile.grid_for(m) for m in cfg.tactile.modalities))
    layouts_by_grid = {
        g: derive_finger_layout(
            urdf_path,
            tuple(finger_link_names),
            probe_grid=tuple(g),
            probe_offset_m=cfg.tactile.probe_offset_m,
            pad_tol_m=cfg.tactile.pad_tol_m,
            info=layout_info,
        )
        for g in grids
    }
    layouts_by_modality = {m: layouts_by_grid[cfg.tactile.grid_for(m)] for m in cfg.tactile.modalities}
    # `bundle.layouts` stays the primary modality's grid: temperature grids and
    # the gate1 calibration use it as "one entry per finger".
    layouts = layouts_by_modality[cfg.tactile.primary_modality]
    # Every object is an indenter for the point-cloud modalities (`elastomer`,
    # `proximity`): their SDF queries run against each tracked link's collision
    # geometry. `_build_options` only forwards this for those two modalities, so
    # it is inert for force/depth (docs §5.2).
    track_link_idx = tuple(int(o.links[0].idx) for o in objects)
    # n_sample_points is a *per-link* budget: an int is split across tracked
    # links (two objects would halve the marker density), so pass one count per
    # link explicitly. `elastomer_params` may still override the key.
    per_link_points = [int(cfg.tactile.proximity_sample_points)] * len(objects)
    elastomer_params = {"n_sample_points": list(per_link_points)}
    elastomer_params.update(cfg.tactile.elastomer_params)
    tactile = build_finger_sensors(
        scene,
        entity=robot,
        layouts_by_modality=layouts_by_modality,
        cfg=cfg.tactile,
        track_link_idx=track_link_idx,
        decimation=cfg.controller.decimation,
        # Sampling density of that point cloud sets the proximity ramp's
        # granularity (docs §9-R22 follow-up).
        extra_params={
            "proximity": {"n_sample_points": list(per_link_points)},
            # Gel-sensor shading parameters come straight from the config (docs §9-R8).
            "elastomer": elastomer_params,
        },
    )
    link_contacts = (
        build_link_contact_sensors(scene, entity=robot, link_names=list(finger_link_names))
        if cfg.tactile.contact_links
        else []
    )

    # Temperature: one shared properties_dict for every sensor in the scene.
    object_links = [o.links[0] for o in objects]
    finger_links = [robot.get_link(n) for n in finger_link_names]
    tc = cfg.temperature
    # All sensors carry base temperature = ambient; each object's grid is
    # re-seeded per episode on top of it. make_properties_dict() takes dicts or
    # ready-made TemperatureProperties and passes the latter through unchanged.
    # Every object shares the file's single material (thermal properties are
    # per-material, and one file holds one material).
    ambient = float(tc.ambient_temp_c)
    properties_dict = make_properties_dict(
        {
            **{
                int(link.idx): properties_from_material(dict(tc.finger_properties), base_temperature_c=ambient)
                for link in finger_links
            },
            **{
                int(link.idx): properties_from_material(
                    dict(cfg.material_properties[material]), base_temperature_c=ambient
                )
                for link in object_links
            },
        },
        default=properties_from_material(dict(tc.environment_properties), base_temperature_c=ambient),
    )

    temperatures = build_finger_temperature_sensors(
        scene,
        entity=robot,
        cfg=tc,
        finger_link_names=list(finger_link_names),
        properties_dict=properties_dict,
    )
    for i, entity in enumerate(objects):
        temperatures.append(
            build_object_temperature_sensor(
                scene,
                entity=entity,
                cfg=tc,
                properties_dict=properties_dict,
                link_name=f"object_{i}",
                # `temperature_object_<i>` is an in-process handle key; the
                # recorded channels under obs/temperature/ keep their names.
                name=f"temperature_object_{i}",
                object_index=i,
            )
        )

    # ------------------------------------------------------------------ cameras
    # Cameras must be declared before build; the eye-in-hand pose can only be
    # resolved afterwards (it needs the built solver's link poses).
    cameras = build_cameras(scene, robot.get_link(hand_link_name), cfg)

    # ------------------------------------------------------------------- build
    scene.build(n_envs=n_envs)

    arm_dofs, finger_dofs, joint_names = _resolve_robot_dofs(robot, cfg)
    robot.set_dofs_kp(np.asarray(cfg.robot.arm_kp, dtype=np.float32), arm_dofs)
    robot.set_dofs_kv(np.asarray(cfg.robot.arm_kv, dtype=np.float32), arm_dofs)
    robot.set_dofs_kp(np.asarray(cfg.robot.finger_kp, dtype=np.float32), finger_dofs)
    robot.set_dofs_kv(np.asarray(cfg.robot.finger_kv, dtype=np.float32), finger_dofs)
    robot.set_dofs_force_range(
        np.asarray(cfg.robot.arm_force_min, dtype=np.float32),
        np.asarray(cfg.robot.arm_force_max, dtype=np.float32),
        arm_dofs,
    )
    robot.set_dofs_force_range(
        np.asarray(cfg.robot.finger_force_min, dtype=np.float32),
        np.asarray(cfg.robot.finger_force_max, dtype=np.float32),
        finger_dofs,
    )

    bundle = SceneBundle(
        cfg=cfg,
        scene=scene,
        robot=robot,
        objects=objects,
        table=table,
        plane=plane,
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
        layout_info=layout_info,
        cameras=cameras,
    )
    bundle.set_robot_init_qpos()
    bundle.sync_cameras()
    return bundle

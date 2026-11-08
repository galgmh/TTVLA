"""场景构建与单集复位：桌面 + 平台 + Panda + 四段式杯子 + 传感器。

本任务是单集单区域受控触碰，所以场景固定为一杯、一平台、一台面：

* 杯子放在 ``object.place_xy`` 指定的槽位，每集加 xy 抖动与偏航抖动；
* 抬升平台与杯体轴线同轴（AABB 中心相对作者原点偏 17.5 mm，构建时按
  实测 AABB 补偿），杯底离桌面 60 mm，否则手掌前缘会撞杯口；
* 不用相机、不用物体侧 TemperatureGrid：杯子四段各是一个热节点，温度
  播种进共享的 ``link_temps`` 表。

构建顺序里有两条硬约束：

1. 所有传感器必须在 ``scene.build()`` **之前**注册；
2. 温度播种必须在 ``scene.reset()`` **之后**、下一帧 ``step()`` 之前。

实体构建位姿与复位位姿是分开的：杯子按槽位位姿构建（传感器在构建时冻结
link 的 AABB，热网格也依赖它），之后每集只按抖动重新摆放。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from sim.config import TaskConfig
from sim.regions import ThermalRegionRuntime, resolve_thermal_regions
from sim.rotations import quat_mul, yaw_to_quat
from sim.sampling.episode_spec import EpisodeSpec
from sim.sensors.finger_layout import FingerLayout, derive_finger_layout
from sim.sensors.tactile import (
    CONTACT_LINKS,
    ELASTOMER_PARAMS,
    PAD_TOL_M,
    PROBE_OFFSET_M,
    PROXIMITY_SAMPLE_POINTS,
    TactileHandle,
    build_finger_sensors,
    build_link_contact_sensors,
    grid_for,
)
from sim.sensors.temperature import (
    TemperatureHandle,
    build_finger_temperature_sensors,
)
from sim.thermal.link_temperature import (
    apply_sensor_variation,
    seed_fingertip_temperatures,
    seed_region_temperatures,
)
from sim.thermal.thermal_materials import build_properties_dict

# 本任务只用一种指尖触觉模态：凝胶标记位移场（估计接触面积）。手指 link 的
# 接触判定与接触力由 build_link_contact_sensors 单独提供。
TACTILE_MODALITIES: tuple[str, ...] = ("elastomer",)


@dataclass(frozen=True)
class CupSpec:
    """杯子在“基础朝向（清单 base_euler_deg）下”的几何事实。"""

    aabb_min: np.ndarray  # (3,) 基础朝向下的 AABB 下界
    aabb_max: np.ndarray  # (3,) 基础朝向下的 AABB 上界
    base_quat: np.ndarray  # (4,) wxyz，资产作者朝向 -> 世界朝向

    @property
    def center(self) -> np.ndarray:
        return 0.5 * (self.aabb_min + self.aabb_max)


def euler_deg_to_quat_wxyz(euler_deg: Sequence[float]) -> tuple[float, float, float, float]:
    """外旋 x-y-z 角度 -> wxyz 四元数（scipy 约定）。

    已用 Genesis 1.4.1 核对：把 z=0..0.30 的瓶子用 ``quat=Rx(+90)`` 构建，
    瓶颈落在世界 -y——与 scipy ``Rotation.from_euler('xyz', (90,0,0))`` 的
    预测完全一致，且 Genesis 的 ``euler`` 参数给出同一位姿到数字位。因此
    全项目只用一种四元数约定。
    """
    from scipy.spatial.transform import Rotation

    q = Rotation.from_euler("xyz", np.asarray(euler_deg, dtype=np.float64), degrees=True).as_quat()
    return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))


def _rotate_aabb(lo: np.ndarray, hi: np.ndarray, quat_wxyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """把 AABB 的 8 个角点按四元数旋转后再取 AABB。"""
    from scipy.spatial.transform import Rotation

    corners = np.array(
        [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])],
        dtype=np.float64,
    )
    xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    rotated = Rotation.from_quat(xyzw).apply(corners)
    return rotated.min(axis=0), rotated.max(axis=0)


def make_cup_spec(object_cfg: dict[str, Any]) -> CupSpec:
    """由资产清单的 AABB 与 ``base_euler_deg`` 造出杯子几何事实。"""
    lo = np.asarray(object_cfg["aabb_min_m"], dtype=np.float64)
    hi = np.asarray(object_cfg["aabb_max_m"], dtype=np.float64)
    base_quat = np.asarray(euler_deg_to_quat_wxyz(object_cfg["base_euler_deg"]), dtype=np.float64)
    # 清单里的 AABB 是作者坐标系下的原始网格；基础朝向旋转后再取 AABB，
    # 才是引擎里“基础朝向物体”的包围盒。
    lo_r, hi_r = _rotate_aabb(lo, hi, base_quat)
    return CupSpec(aabb_min=lo_r, aabb_max=hi_r, base_quat=base_quat)


def place_cup(
    cfg: TaskConfig,
    spec: CupSpec,
    *,
    xy_offset: Sequence[float] = (0.0, 0.0),
    yaw: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """杯子在世界系里的位姿 ``(pos, wxyz quat)``。

    AABB 中心立在槽位上方，底面贴平台顶面，偏航绕该中心旋转。``xy_offset``
    是相对槽位的每集抖动。
    """
    from scipy.spatial.transform import Rotation

    table = cfg["table"]
    slot = np.asarray(cfg["object"]["place_xy"], dtype=np.float64) + np.asarray(xy_offset, dtype=np.float64)
    yaw_quat = yaw_to_quat(float(yaw))
    quat = quat_mul(yaw_quat, spec.base_quat)
    # scipy 收 xyzw，本项目的四元数是 wxyz。
    offset = Rotation.from_quat([yaw_quat[1], yaw_quat[2], yaw_quat[3], yaw_quat[0]]).apply(spec.center)
    pos = np.array(
        [
            float(table["pos"][0]) + float(slot[0]) - offset[0],
            float(table["pos"][1]) + float(slot[1]) - offset[1],
            float(table["height_m"]) + float(cfg["object"]["rest_height_m"]) - spec.aabb_min[2],
        ],
        dtype=np.float64,
    )
    return pos, quat


@dataclass
class SceneBundle:
    """构建好的场景与所有句柄。"""

    cfg: TaskConfig
    scene: Any
    robot: Any
    cup: Any
    cup_spec: CupSpec
    regions: ThermalRegionRuntime
    layouts: list[FingerLayout]
    tactile_layouts: dict[str, list[FingerLayout]]
    tactile: list[TactileHandle]
    link_contacts: list[TactileHandle]
    temperatures: list[TemperatureHandle]
    arm_dofs: np.ndarray
    finger_dofs: np.ndarray
    hand_link_name: str
    finger_link_names: tuple[str, str]
    joint_names: list[str] = field(default_factory=list)
    link_names: list[str] = field(default_factory=list)
    n_envs: int = 1

    # --------------------------------------------------------------- 状态读取
    def hand_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """手 link 的世界位姿 (pos, wxyz quat)。"""
        link = self.robot.get_link(self.hand_link_name)
        pos = np.asarray(link.get_pos().detach().cpu()).reshape(-1)[:3]
        quat = np.asarray(link.get_quat().detach().cpu()).reshape(-1)[:4]
        return pos.astype(np.float64), quat.astype(np.float64)

    def cup_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """杯子的世界位姿 (pos, wxyz quat)。"""
        pos = np.asarray(self.cup.get_pos().detach().cpu()).reshape(-1)[:3]
        quat = np.asarray(self.cup.get_quat().detach().cpu()).reshape(-1)[:4]
        return pos.astype(np.float64), quat.astype(np.float64)

    def joint_state(self) -> np.ndarray:
        """7 个臂关节 + 2 个指关节的位置。"""
        dofs = np.concatenate([self.arm_dofs, self.finger_dofs])
        q = np.asarray(self.robot.get_dofs_position(dofs_idx_local=dofs).detach().cpu())
        return q.reshape(-1).astype(np.float64)

    def finger_positions(self) -> np.ndarray:
        """两个指关节的位置 (2,)，单位米。"""
        q = np.asarray(self.robot.get_dofs_position(dofs_idx_local=self.finger_dofs).detach().cpu())
        return q.reshape(-1).astype(np.float64)

    def gripper_width(self) -> float:
        """夹爪开口 = 两个指关节位置之和（各 0..0.04 m）。"""
        return float(np.sum(self.finger_positions()))

    # ------------------------------------------------------------------ 控制
    def step(self) -> None:
        self.scene.step()

    def set_robot_init_qpos(self) -> None:
        q = np.asarray(self.cfg["robot"]["init_qpos"], dtype=np.float64)
        self.robot.set_dofs_position(q, zero_velocity=True)
        self.robot.control_dofs_position(q)

    # ---------------------------------------------------------------- 单集复位
    def reset_episode(self, spec: EpisodeSpec) -> None:
        """reset -> 机器人初始位姿 -> 杯子位姿 -> 传感器参数 -> 温度播种。

        ``spec`` 是采样器给出的不可变 ``EpisodeSpec``。播种顺序是硬约束：
        ``scene.reset()`` 会把每个 link 的温度写回基准温度，因此所有播种都
        必须落在 reset 之后、下一次 ``step()`` 之前。
        """
        self.scene.reset()
        self.set_robot_init_qpos()

        pos, quat = place_cup(
            self.cfg,
            self.cup_spec,
            xy_offset=spec.cup_xy_jitter_m,
            yaw=spec.cup_yaw_jitter_rad,
        )
        self.cup.set_pos(pos)
        self.cup.set_quat(quat)

        # 传感器非理想性：每集按 spec 重设（公共 setter + RC 的元数据直写）。
        apply_sensor_variation(
            self.temperatures,
            rc_s=spec.sensor_rc_s,
            noise_c=spec.sensor_noise_c,
            bias_c=spec.sensor_bias_c,
            quantization_c=spec.sensor_quantization_c,
        )
        # 指尖初始温度：左右各一个值（70% 常温、30% 上一段接触的余温）；
        # 真值与测量通道一起初始化，读数从第一帧起就是该值。
        seed_fingertip_temperatures(self.temperatures, list(spec.fingertip_temps_c))
        # 杯子四段的初始温度，按 region_order 顺序。
        seed_region_temperatures(
            self.temperatures[0], self.regions, list(spec.link_temps_c)
        )


def _resolve_robot_dofs(robot, cfg: TaskConfig) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """把关节名解析成 dof 下标（机器人实体内的局部下标）。"""

    def _one(name: str) -> int:
        joint = robot.get_joint(name=name)
        idx = list(joint.dofs_idx_local)
        if len(idx) != 1:
            raise ValueError(f"关节 {name!r} 应只有一个自由度，实为 {idx}")
        return int(idx[0])

    robot_cfg = cfg["robot"]
    arm = [_one(n) for n in robot_cfg["arm_joint_names"]]
    fingers = [_one(n) for n in robot_cfg["finger_joint_names"]]
    names = list(robot_cfg["arm_joint_names"]) + list(robot_cfg["finger_joint_names"])
    return np.asarray(arm, dtype=np.int64), np.asarray(fingers, dtype=np.int64), names


def _resolve_links(robot, hand_name: str, finger_names: Sequence[str]) -> tuple[str, tuple[str, str], list[str]]:
    """确认手 link 与两个指 link 存在，返回 (手名, 指名元组, 全部 link 名)。"""
    have = [str(link.name) for link in robot.links]
    for name in (hand_name, *finger_names):
        if name not in have:
            raise ValueError(f"机器人缺少 link {name!r}；可用：{have}")
    return str(hand_name), (str(finger_names[0]), str(finger_names[1])), have


def build_scene(
    cfg: TaskConfig,
    seed: int = 0,
    n_envs: int = 1,
    show_viewer: bool = False,
    logging_level: str = "warning",
) -> SceneBundle:
    """按配置构建场景（单杯、单平台、无相机）。"""
    import genesis as gs

    from sim.compat.sensor_compat import install_contact_area_policy, install_sensor_reset_fix
    from sim.compat.mesh_load_compat import install_texture_resolver_policy

    if not gs._initialized:
        gs.init(backend=gs.gpu, precision="32", seed=seed, logging_level=logging_level)

    # 传感器 reset 补丁 + 纹理解析补丁（幂等）。
    install_sensor_reset_fix()
    install_texture_resolver_policy()

    object_cfg = cfg["object"]
    sensor_cfg = cfg["sensor"]
    robot_cfg = cfg["robot"]
    table_cfg = cfg["table"]
    platform_cfg = cfg["platform"]

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
        sim_options=gs.options.SimOptions(dt=float(cfg["controller"]["sim_dt"]), substeps=1),
        rigid_options=rigid_options,
        show_viewer=show_viewer,
    )

    # ------------------------------------------------------------------ 桌面
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(
        gs.morphs.Box(
            size=tuple(table_cfg["size"]),
            pos=(
                float(table_cfg["pos"][0]),
                float(table_cfg["pos"][1]),
                float(table_cfg["height_m"]) - 0.5 * float(table_cfg["size"][2]),
            ),
            fixed=True,
        ),
        surface=gs.surfaces.Default(color=tuple(table_cfg["color"])),
    )

    # ------------------------------------------------------------------ 机器人
    robot_urdf = Path(robot_cfg["urdf"])
    if not robot_urdf.is_absolute():
        from sim.config import REPO_ROOT

        robot_urdf = REPO_ROOT / robot_urdf
    if not robot_urdf.is_file():
        raise FileNotFoundError(f"机器人 URDF 不存在：{robot_urdf}")
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=str(robot_urdf.resolve()),
            pos=tuple(robot_cfg["base_pos"]),
            quat=tuple(robot_cfg["base_quat"]),
            fixed=True,
            merge_fixed_links=bool(robot_cfg["merge_fixed_links"]),
        )
    )

    # -------------------------------------------------------------------- 杯子
    cup_spec = make_cup_spec(object_cfg)
    nominal_pos, nominal_quat = place_cup(cfg, cup_spec)
    cup_urdf = Path(object_cfg["urdf_path"])
    if not cup_urdf.is_file():
        raise FileNotFoundError(f"杯子 URDF 不存在：{cup_urdf}")
    cup = scene.add_entity(
        gs.morphs.URDF(
            file=str(cup_urdf.resolve()),
            pos=tuple(float(v) for v in nominal_pos),
            quat=tuple(float(v) for v in nominal_quat),
            fixed=False,
            merge_fixed_links=False,
            # align=False 是关键：默认值会把根 link 重定位到质心，之后所有
            # 基于作者坐标系的摆放与热区映射都会偏。
            align=False,
            convexify=True,
        ),
        material=gs.materials.Rigid(friction=1.0),
    )
    regions = resolve_thermal_regions(cup, object_cfg)

    # ------------------------------------------------------------------ 平台
    # 平台与杯体轴线同轴：AABB 中心相对作者原点偏 17.5 mm（手柄侧），
    # 所以平台中心 = 槽位 − AABB 中心（基础朝向下）。
    center = cup_spec.center
    platform_pos = (
        float(table_cfg["pos"][0]) + float(object_cfg["place_xy"][0]) - float(center[0]),
        float(table_cfg["pos"][1]) + float(object_cfg["place_xy"][1]) - float(center[1]),
        float(table_cfg["height_m"]) + 0.5 * float(platform_cfg["height_m"]),
    )
    scene.add_entity(
        gs.morphs.Box(
            size=(
                float(platform_cfg["size_xy"][0]),
                float(platform_cfg["size_xy"][1]),
                float(platform_cfg["height_m"]),
            ),
            pos=platform_pos,
            fixed=True,
        ),
        surface=gs.surfaces.Default(color=tuple(platform_cfg["color"])),
    )

    # ------------------------------------------------------------------ 触觉
    hand_link_name, finger_link_names, link_names = _resolve_links(
        robot, str(robot_cfg["hand_link_name"]), list(robot_cfg["finger_link_names"])
    )
    grids = list(dict.fromkeys(grid_for(m) for m in TACTILE_MODALITIES))
    layouts_by_grid = {
        g: derive_finger_layout(
            robot_urdf,
            tuple(finger_link_names),
            probe_grid=tuple(g),
            probe_offset_m=PROBE_OFFSET_M,
            pad_tol_m=PAD_TOL_M,
        )
        for g in grids
    }
    layouts_by_modality = {m: layouts_by_grid[grid_for(m)] for m in TACTILE_MODALITIES}
    # bundle.layouts 是（唯一）触觉模态的布局：接触面积、接触方向都用它。
    layouts = layouts_by_modality[TACTILE_MODALITIES[0]]

    cup_link_indices = tuple(int(i) for i in regions.global_link_indices)
    per_link_points = [int(PROXIMITY_SAMPLE_POINTS)] * len(cup_link_indices)
    elastomer_params = {"n_sample_points": list(per_link_points)}
    elastomer_params.update(ELASTOMER_PARAMS)
    tactile = build_finger_sensors(
        scene,
        entity=robot,
        layouts_by_modality=layouts_by_modality,
        modalities=TACTILE_MODALITIES,
        track_link_idx=cup_link_indices,
        decimation=int(cfg["controller"]["decimation"]),
        extra_params={"elastomer": elastomer_params},
    )
    link_contacts = (
        build_link_contact_sensors(scene, entity=robot, link_names=list(finger_link_names))
        if CONTACT_LINKS
        else []
    )

    # ------------------------------------------------------------------ 温度
    finger_links = [robot.get_link(n) for n in finger_link_names]
    ambient = float(sensor_cfg["ambient_temp_c"])
    properties_dict = build_properties_dict(
        object_cfg=object_cfg,
        sensor_cfg=sensor_cfg,
        finger_links=finger_links,
        cup_link_indices=cup_link_indices,
    )
    temperatures = build_finger_temperature_sensors(
        scene,
        entity=robot,
        finger_link_names=list(finger_link_names),
        grid_size=tuple(int(v) for v in sensor_cfg["finger_grid"]),
        ambient_temp_c=ambient,
        nominal=dict(sensor_cfg["nominal"]),
        properties_dict=properties_dict,
    )

    # 接触面积策略（pad_face）：指尖接触面积改写为 A_i = A_face/sqrt(n_i)，
    # 让一次触碰的两个手指在各接触阶段拿到同样的接触热流。
    finger_faces = {
        int(link.idx): layout.pad_face_area_m2
        for link, layout in zip(finger_links, layouts)
    }
    install_contact_area_policy(finger_faces)
    faces = " ".join(
        f"{name} {area * 1e6:.3f}" for name, area in zip(finger_link_names, finger_faces.values())
    )
    print(f"  接触面积策略 pad_face：{faces} mm^2 -> A_i = A/sqrt(n_i)，sum(A_i^2) = A^2")

    # ------------------------------------------------------------------ 构建
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
        cup=cup,
        cup_spec=cup_spec,
        regions=regions,
        layouts=layouts,
        tactile_layouts=layouts_by_modality,
        tactile=tactile,
        link_contacts=link_contacts,
        temperatures=temperatures,
        arm_dofs=arm_dofs,
        finger_dofs=finger_dofs,
        hand_link_name=hand_link_name,
        finger_link_names=finger_link_names,
        joint_names=joint_names,
        link_names=link_names,
        n_envs=n_envs,
    )
    bundle.set_robot_init_qpos()
    return bundle


__all__ = [
    "CupSpec",
    "SceneBundle",
    "TACTILE_MODALITIES",
    "build_scene",
    "euler_deg_to_quat_wxyz",
    "make_cup_spec",
    "place_cup",
]

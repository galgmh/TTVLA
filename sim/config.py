"""配置加载：YAML -> 经过严格校验的普通嵌套字典。

本项目把配置拆成五类文件，各自只描述一件事：

* ``configs/objects/*.yaml``    物体资产引用 + 放置方式 + 热物性
* ``configs/sensors/*.yaml``    指尖温度传感器的网格与环境温度
* ``configs/tasks/*.yaml``      机器人、控制器、桌面、探测区域与时序
* ``configs/datasets/*.yaml``   数据采样规则、划分与输出路径
* ``configs/models/*.yaml``     热编码器结构与训练超参

任务配置通过 ``object`` / ``sensor`` 两个键引用前两类文件；加载时被引用
文件的内容会直接展开进返回的字典（``cfg["object"]`` / ``cfg["sensor"]``），
所以下游模块只面对一个完整的字典。

本模块只依赖标准库与 PyYAML，不导入 Genesis / torch——测试可以脱离仿真运行。
所有未知键都会在加载时被拒绝：配置拼写错误必须立刻暴露，而不是运行到一半
才在某个模块里 ``KeyError``。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

# 仓库根目录：配置里的相对路径（assets/...、configs/...）都以它为基准。
REPO_ROOT = Path(__file__).resolve().parents[1]

# 任务配置：嵌套的普通 dict，与 YAML 的写法一一对应。
TaskConfig = dict[str, Any]


# ---------------------------------------------------------------- 通用校验工具


def _check_keys(
    data: dict[str, Any],
    *,
    required: frozenset[str],
    optional: frozenset[str],
    where: str,
) -> None:
    """键集合必须恰好是 required ∪ optional；多余和缺失都要报错。"""
    if not isinstance(data, dict):
        raise ValueError(f"{where} 必须是映射（dict），实际是 {type(data).__name__}")
    missing = required - data.keys()
    unknown = data.keys() - required - optional
    if missing:
        raise ValueError(f"{where} 缺少键 {sorted(missing)}")
    if unknown:
        raise ValueError(f"{where} 出现未知键 {sorted(unknown)}")


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _num(value: Any, where: str) -> float:
    if not _is_num(value):
        raise ValueError(f"{where} 必须是数值，实际是 {value!r}")
    return float(value)


def _int(value: Any, where: str, *, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{where} 必须是整数，实际是 {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{where} 必须 >= {minimum}，实际是 {value!r}")
    return value


def _num_list(value: Any, where: str, length: int) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{where} 必须是长度 {length} 的数值列表，实际是 {value!r}")
    return [_num(v, f"{where}[{i}]") for i, v in enumerate(value)]


def _range(value: Any, where: str, *, positive: bool = True) -> list[float]:
    lo, hi = _num_list(value, where, 2)
    if lo > hi:
        raise ValueError(f"{where} 必须满足 low <= high，实际是 {value!r}")
    if positive and lo <= 0.0:
        raise ValueError(f"{where} 必须为正，实际是 {value!r}")
    return [lo, hi]


def _str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where} 必须是非空字符串，实际是 {value!r}")
    return value


def _str_list(value: Any, where: str, *, length: int | None = None) -> list[str]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{where} 必须是非空字符串列表，实际是 {value!r}")
    if length is not None and len(value) != length:
        raise ValueError(f"{where} 必须包含 {length} 项，实际是 {value!r}")
    if not all(isinstance(v, str) and v for v in value):
        raise ValueError(f"{where} 只能包含非空字符串，实际是 {value!r}")
    return list(value)


def _check_color(value: Any, where: str) -> None:
    if not isinstance(value, (list, tuple)) or len(value) not in (3, 4):
        raise ValueError(f"{where} 必须是 3 或 4 元 RGB(A) 列表，实际是 {value!r}")
    for i, v in enumerate(value):
        _num(v, f"{where}[{i}]")


def _load_yaml(path: str | Path, what: str) -> dict[str, Any]:
    """读入一个 YAML 文件，必须是顶层映射。"""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{what}不存在：{path}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{what}的顶层必须是映射：{path}")
    return raw


def _repo_path(path: str | Path) -> Path:
    """配置里的相对路径一律相对仓库根目录解析。"""
    p = Path(path).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p


# ------------------------------------------------------------------ 物体配置

_OBJECT_REQUIRED = frozenset(
    {
        "manifest",
        "material",
        "material_properties",
        "place_xy",
        "rest_height_m",
        "xy_jitter_m",
        "yaw_jitter_rad",
    }
)
_OBJECT_OPTIONAL = frozenset({"color"})
_MATERIAL_KEYS = frozenset({"conductivity", "density", "specific_heat", "emissivity"})
# 资产清单（manifest）由 scripts/build_cup_4link.py 生成，是区域顺序、link
# 名称和区域 z 边界的唯一来源；物体配置只引用它，不再重复抄写。
_MANIFEST_REQUIRED = frozenset(
    {
        "asset_version",
        "urdf_file",
        "region_order",
        "regions",
        "link_names",
        "base_euler_deg",
        "mass_kg",
        "aabb_min_m",
        "aabb_max_m",
    }
)


def _validate_manifest(manifest: dict[str, Any], source: str) -> None:
    _check_keys(
        manifest,
        required=_MANIFEST_REQUIRED,
        optional=frozenset({"meshes", "notes"}),
        where=f"{source} (manifest)",
    )
    _str(manifest["asset_version"], f"{source}: asset_version")
    _str(manifest["urdf_file"], f"{source}: urdf_file")
    order = _str_list(manifest["region_order"], f"{source}: region_order")
    if len(set(order)) != len(order):
        raise ValueError(f"{source}: region_order 不能重复")
    if len(order) != 4:
        raise ValueError(f"{source}: 本任务固定四个热区，region_order 必须恰好 4 项")
    link_names = _str_list(manifest["link_names"], f"{source}: link_names")
    if set(link_names) != set(order):
        raise ValueError(f"{source}: link_names 与 region_order 必须一致（本资产一个区域对应一个 link）")
    regions = manifest["regions"]
    if not isinstance(regions, dict) or set(regions) != set(order):
        raise ValueError(f"{source}: regions 的键必须恰好是 region_order")
    for name in order:
        entry = regions[name]
        _check_keys(
            entry,
            required=frozenset({"link", "z_span_m"}),
            optional=frozenset(),
            where=f"{source}: regions.{name}",
        )
        link = _str(entry["link"], f"{source}: regions.{name}.link")
        if link not in link_names:
            raise ValueError(f"{source}: regions.{name}.link = {link!r} 不在 link_names 中")
        lo, hi = _num_list(entry["z_span_m"], f"{source}: regions.{name}.z_span_m", 2)
        if not lo < hi:
            raise ValueError(f"{source}: regions.{name}.z_span_m 必须满足 low < high")
    _num_list(manifest["base_euler_deg"], f"{source}: base_euler_deg", 3)
    if _num(manifest["mass_kg"], f"{source}: mass_kg") <= 0.0:
        raise ValueError(f"{source}: mass_kg 必须为正")
    _num_list(manifest["aabb_min_m"], f"{source}: aabb_min_m", 3)
    _num_list(manifest["aabb_max_m"], f"{source}: aabb_max_m", 3)


def load_object_config(path: str | Path) -> dict[str, Any]:
    """加载物体配置，并把 manifest 的资产字段展开进来。"""
    source = str(path)
    cfg = _load_yaml(path, "物体配置")
    _check_keys(cfg, required=_OBJECT_REQUIRED, optional=_OBJECT_OPTIONAL, where=source)

    manifest_path = _repo_path(_str(cfg["manifest"], f"{source}: manifest"))
    manifest = _load_yaml(manifest_path, "资产清单")
    _validate_manifest(manifest, str(manifest_path))

    props = cfg["material_properties"]
    if not isinstance(props, dict) or not props:
        raise ValueError(f"{source}: material_properties 必须是非空映射")
    for name, values in props.items():
        _check_keys(
            values,
            required=_MATERIAL_KEYS,
            optional=frozenset(),
            where=f"{source}: material_properties.{name}",
        )
        for key in _MATERIAL_KEYS:
            _num(values[key], f"{source}: material_properties.{name}.{key}")
    material = _str(cfg["material"], f"{source}: material")
    if material not in props:
        raise ValueError(f"{source}: material = {material!r} 不在 material_properties 中")

    _num_list(cfg["place_xy"], f"{source}: place_xy", 2)
    if _num(cfg["rest_height_m"], f"{source}: rest_height_m") < 0.0:
        raise ValueError(f"{source}: rest_height_m 必须 >= 0")
    for key in ("xy_jitter_m", "yaw_jitter_rad"):
        if _num(cfg[key], f"{source}: {key}") < 0.0:
            raise ValueError(f"{source}: {key} 必须 >= 0")
    if "color" in cfg:
        _check_color(cfg["color"], f"{source}: color")

    # 资产字段直接并入：下游只需 cfg["object"]["region_order"] 等。
    merged = dict(cfg)
    merged["manifest_path"] = str(manifest_path)
    merged["urdf_path"] = str(manifest_path.parent / manifest["urdf_file"])
    merged.update(
        {
            "asset_version": manifest["asset_version"],
            "region_order": list(manifest["region_order"]),
            "region_links": {n: manifest["regions"][n]["link"] for n in manifest["region_order"]},
            "region_z_spans": {n: list(manifest["regions"][n]["z_span_m"]) for n in manifest["region_order"]},
            "link_names": list(manifest["link_names"]),
            "base_euler_deg": list(manifest["base_euler_deg"]),
            "mass_kg": float(manifest["mass_kg"]),
            "aabb_min_m": list(manifest["aabb_min_m"]),
            "aabb_max_m": list(manifest["aabb_max_m"]),
        }
    )
    return merged


# ------------------------------------------------------------------ 传感器配置

_SENSOR_REQUIRED = frozenset(
    {
        "finger_grid",
        "object_grid",
        "ambient_temp_c",
        "nominal",
    }
)


def load_sensor_config(path: str | Path) -> dict[str, Any]:
    """加载指尖温度传感器配置。"""
    source = str(path)
    cfg = _load_yaml(path, "传感器配置")
    _check_keys(cfg, required=_SENSOR_REQUIRED, optional=frozenset(), where=source)

    grid = cfg["finger_grid"]
    if not isinstance(grid, (list, tuple)) or len(grid) != 3:
        raise ValueError(f"{source}: finger_grid 必须是 3 个正整数")
    for i, v in enumerate(grid):
        _int(v, f"{source}: finger_grid[{i}]", minimum=1)
    if cfg["object_grid"] is not None:
        raise ValueError(f"{source}: object_grid 必须为 null（本任务的物体热状态是每 link 一个集总温度）")
    _num(cfg["ambient_temp_c"], f"{source}: ambient_temp_c")
    nominal = cfg["nominal"]
    _check_keys(
        nominal,
        required=frozenset({"rc_s", "noise_c", "bias_c", "quantization_c"}),
        optional=frozenset(),
        where=f"{source}: nominal",
    )
    for key in ("rc_s", "noise_c", "bias_c", "quantization_c"):
        _num(nominal[key], f"{source}: nominal.{key}")
    if float(nominal["rc_s"]) < 0.0 or float(nominal["noise_c"]) < 0.0 or float(nominal["quantization_c"]) < 0.0:
        raise ValueError(f"{source}: nominal 的 rc/noise/quantization 必须 >= 0")
    return cfg


# --------------------------------------------------------------------- 任务配置

_TASK_REQUIRED = frozenset(
    {
        "task_id",
        "seed",
        "object",
        "sensor",
        "robot",
        "controller",
        "table",
        "platform",
        "regions",
        "mixed_probe",
        "timing",
        "contact",
    }
)

_ROBOT_KEYS = frozenset(
    {
        "urdf",
        "merge_fixed_links",
        "hand_link_name",
        "finger_link_names",
        "arm_joint_names",
        "finger_joint_names",
        "tcp_offset",
        "open_finger_pos",
        "closed_finger_pos",
        "base_pos",
        "base_quat",
        "init_qpos",
        "arm_kp",
        "arm_kv",
        "finger_kp",
        "finger_kv",
        "arm_force_min",
        "arm_force_max",
        "finger_force_min",
        "finger_force_max",
    }
)
_CONTROLLER_KEYS = frozenset(
    {
        "pos_scale_m",
        "rot_scale_rad",
        "ik_damping",
        "ik_max_iters",
        "ik_max_samples",
        "ik_max_step_rad",
        "pos_speed_m_per_s",
        "rot_speed_rad_per_s",
        "control_hz",
        "sim_dt",
        "decimation",
        "finger_speed_m_per_s",
    }
)
_PROBE_KEYS = frozenset({"pos", "tool_dir", "finger_dir", "pre_dist"})
_PROBE_OPTIONAL = frozenset({"pre_grip", "xy_jitter_m", "z_jitter_m"})


def _validate_probe(entry: dict[str, Any], where: str) -> None:
    _check_keys(entry, required=_PROBE_KEYS, optional=_PROBE_OPTIONAL, where=where)
    _num_list(entry["pos"], f"{where}.pos", 3)
    tool = _num_list(entry["tool_dir"], f"{where}.tool_dir", 3)
    finger = _num_list(entry["finger_dir"], f"{where}.finger_dir", 3)
    tool_norm = math.sqrt(sum(v * v for v in tool))
    finger_norm = math.sqrt(sum(v * v for v in finger))
    if tool_norm <= 1e-9:
        raise ValueError(f"{where}.tool_dir 不能是零向量")
    if finger_norm <= 1e-9:
        raise ValueError(f"{where}.finger_dir 不能是零向量")
    cross = (
        tool[1] * finger[2] - tool[2] * finger[1],
        tool[2] * finger[0] - tool[0] * finger[2],
        tool[0] * finger[1] - tool[1] * finger[0],
    )
    if math.sqrt(sum(v * v for v in cross)) <= 1e-6 * tool_norm * finger_norm:
        raise ValueError(f"{where}.finger_dir 不能与 tool_dir 平行")
    if _num(entry["pre_dist"], f"{where}.pre_dist") <= 0.0:
        raise ValueError(f"{where}.pre_dist 必须为正")
    if "pre_grip" in entry:
        grip = _num(entry["pre_grip"], f"{where}.pre_grip")
        if not -1.0 <= grip <= 1.0:
            raise ValueError(f"{where}.pre_grip 必须在 [-1, 1]")
    for key in ("xy_jitter_m", "z_jitter_m"):
        if key in entry and _num(entry[key], f"{where}.{key}") < 0.0:
            raise ValueError(f"{where}.{key} 必须 >= 0")


def _validate_task(cfg: dict[str, Any], source: str) -> None:
    _check_keys(cfg, required=_TASK_REQUIRED, optional=frozenset(), where=source)
    _str(cfg["task_id"], f"{source}: task_id")
    _int(cfg["seed"], f"{source}: seed")
    _str(cfg["object"], f"{source}: object")
    _str(cfg["sensor"], f"{source}: sensor")

    robot = cfg["robot"]
    _check_keys(robot, required=_ROBOT_KEYS, optional=frozenset(), where=f"{source}: robot")
    _str(robot["urdf"], f"{source}: robot.urdf")
    if not isinstance(robot["merge_fixed_links"], bool):
        raise ValueError(f"{source}: robot.merge_fixed_links 必须是布尔值")
    _str(robot["hand_link_name"], f"{source}: robot.hand_link_name")
    _str_list(robot["finger_link_names"], f"{source}: robot.finger_link_names", length=2)
    arm = _str_list(robot["arm_joint_names"], f"{source}: robot.arm_joint_names")
    fingers = _str_list(robot["finger_joint_names"], f"{source}: robot.finger_joint_names", length=2)
    _num_list(robot["tcp_offset"], f"{source}: robot.tcp_offset", 3)
    _num(robot["open_finger_pos"], f"{source}: robot.open_finger_pos")
    _num(robot["closed_finger_pos"], f"{source}: robot.closed_finger_pos")
    if float(robot["closed_finger_pos"]) >= float(robot["open_finger_pos"]):
        raise ValueError(f"{source}: robot.closed_finger_pos 必须小于 open_finger_pos")
    _num_list(robot["base_pos"], f"{source}: robot.base_pos", 3)
    _num_list(robot["base_quat"], f"{source}: robot.base_quat", 4)
    qpos = robot["init_qpos"]
    if not isinstance(qpos, (list, tuple)) or len(qpos) != len(arm) + len(fingers):
        raise ValueError(f"{source}: robot.init_qpos 必须有 {len(arm) + len(fingers)} 项")
    for i, v in enumerate(qpos):
        _num(v, f"{source}: robot.init_qpos[{i}]")
    for key in ("arm_kp", "arm_kv", "arm_force_min", "arm_force_max"):
        _num_list(robot[key], f"{source}: robot.{key}", len(arm))
    for key in ("finger_kp", "finger_kv", "finger_force_min", "finger_force_max"):
        _num_list(robot[key], f"{source}: robot.{key}", len(fingers))

    controller = cfg["controller"]
    _check_keys(controller, required=_CONTROLLER_KEYS, optional=frozenset(), where=f"{source}: controller")
    for key in ("pos_scale_m", "rot_scale_rad", "ik_damping", "control_hz", "sim_dt",
                "ik_max_step_rad", "pos_speed_m_per_s", "rot_speed_rad_per_s"):
        if _num(controller[key], f"{source}: controller.{key}") <= 0.0:
            raise ValueError(f"{source}: controller.{key} 必须为正")
    for key in ("ik_max_iters", "ik_max_samples", "decimation"):
        _int(controller[key], f"{source}: controller.{key}", minimum=1)
    if _num(controller["finger_speed_m_per_s"], f"{source}: controller.finger_speed_m_per_s") < 0.0:
        raise ValueError(f"{source}: controller.finger_speed_m_per_s 必须 >= 0")
    # 采样率必须能整除仿真步：100 Hz 采样对应每 2 个 5 ms 子步读一次。
    ratio = 1.0 / float(controller["sim_dt"]) / (float(controller["control_hz"]) * int(controller["decimation"]))
    if abs(ratio - 1.0) > 1e-6:
        raise ValueError(f"{source}: sim_dt * control_hz * decimation 必须等于 1")

    table = cfg["table"]
    _check_keys(
        table,
        required=frozenset({"pos", "size", "height_m"}),
        optional=frozenset({"color"}),
        where=f"{source}: table",
    )
    _num_list(table["pos"], f"{source}: table.pos", 3)
    size = _num_list(table["size"], f"{source}: table.size", 3)
    if any(v <= 0.0 for v in size):
        raise ValueError(f"{source}: table.size 必须为正")
    _num(table["height_m"], f"{source}: table.height_m")
    if "color" in table:
        _check_color(table["color"], f"{source}: table.color")

    platform = cfg["platform"]
    _check_keys(
        platform,
        required=frozenset({"size_xy", "height_m", "material"}),
        optional=frozenset({"color"}),
        where=f"{source}: platform",
    )
    _num_list(platform["size_xy"], f"{source}: platform.size_xy", 2)
    if _num(platform["height_m"], f"{source}: platform.height_m") <= 0.0:
        raise ValueError(f"{source}: platform.height_m 必须为正")
    _str(platform["material"], f"{source}: platform.material")
    if "color" in platform:
        _check_color(platform["color"], f"{source}: platform.color")

    regions = cfg["regions"]
    if not isinstance(regions, dict) or not regions:
        raise ValueError(f"{source}: regions 必须是非空映射")
    for name, entry in regions.items():
        _validate_probe(entry, f"{source}: regions.{name}")
    _validate_probe(cfg["mixed_probe"], f"{source}: mixed_probe")

    timing = cfg["timing"]
    _check_keys(
        timing,
        required=frozenset({"contact_s", "post_contact_s"}),
        optional=frozenset(),
        where=f"{source}: timing",
    )
    for key in ("contact_s", "post_contact_s"):
        if _num(timing[key], f"{source}: timing.{key}") < 0.0:
            raise ValueError(f"{source}: timing.{key} 必须 >= 0")

    contact = cfg["contact"]
    _check_keys(
        contact,
        required=frozenset(
            {
                "nominal_force_n",
                "force_multipliers",
                "force_band_n",
                "seek_timeout_s",
                "close_step_m",
                "gain_m_per_n",
                "brief_hold_s",
                "fail_retreat_m",
                "sample_every_sim_steps",
            }
        ),
        optional=frozenset(),
        where=f"{source}: contact",
    )
    if _num(contact["nominal_force_n"], f"{source}: contact.nominal_force_n") <= 0.0:
        raise ValueError(f"{source}: contact.nominal_force_n 必须为正")
    multipliers = contact["force_multipliers"]
    if not isinstance(multipliers, (list, tuple)) or not multipliers:
        raise ValueError(f"{source}: contact.force_multipliers 必须是非空列表")
    for i, v in enumerate(multipliers):
        if _num(v, f"{source}: contact.force_multipliers[{i}]") <= 0.0:
            raise ValueError(f"{source}: contact.force_multipliers 必须为正")
    for key in ("force_band_n", "seek_timeout_s", "close_step_m", "gain_m_per_n"):
        if _num(contact[key], f"{source}: contact.{key}") <= 0.0:
            raise ValueError(f"{source}: contact.{key} 必须为正")
    _range(contact["brief_hold_s"], f"{source}: contact.brief_hold_s", positive=False)
    _range(contact["fail_retreat_m"], f"{source}: contact.fail_retreat_m")
    _int(contact["sample_every_sim_steps"], f"{source}: contact.sample_every_sim_steps", minimum=1)


def load_task_config(path: str | Path) -> TaskConfig:
    """加载任务配置，并把引用的物体/传感器配置展开进来。"""
    cfg = _load_yaml(path, "任务配置")
    _validate_task(cfg, str(path))
    cfg = dict(cfg)
    object_path = cfg["object"]
    sensor_path = cfg["sensor"]
    cfg["object"] = load_object_config(_repo_path(object_path))
    cfg["sensor"] = load_sensor_config(_repo_path(sensor_path))
    # 跨文件一致性：探测区域必须与资产清单的热区一一对应，平台材质必须存在。
    order = set(cfg["object"]["region_order"])
    if set(cfg["regions"]) != order:
        raise ValueError(
            f"{path}: regions 的键 {sorted(cfg['regions'])} 必须与资产清单的 "
            f"region_order {sorted(order)} 一致"
        )
    platform_material = cfg["platform"]["material"]
    if platform_material not in cfg["object"]["material_properties"]:
        raise ValueError(f"{path}: platform.material = {platform_material!r} 不在物体配置的热物性表中")
    # 数据生成器需要知道任务/物体/传感器配置文件的位置，写进元数据。
    cfg["config_paths"] = {
        "task": str(Path(path)),
        "object": cfg["object"]["manifest_path"],
        "sensor": str(_repo_path(sensor_path)),
    }
    return cfg


# ------------------------------------------------------------------- 数据集配置

_DATASET_REQUIRED = frozenset(
    {
        "dataset_version",
        "output_dir",
        "seed",
        "sample_hz",
        "training_hz",
        "prefixes_s",
        "region_temps_c",
        "fingertip",
        "force_levels",
        "contact_quality_probabilities",
        "sensor",
        "splits",
    }
)


def load_dataset_config(path: str | Path) -> dict[str, Any]:
    """加载 ThermalContact-v1 数据采样配置。"""
    source = str(path)
    cfg = _load_yaml(path, "数据集配置")
    _check_keys(cfg, required=_DATASET_REQUIRED, optional=frozenset(), where=source)
    _str(cfg["dataset_version"], f"{source}: dataset_version")
    _str(cfg["output_dir"], f"{source}: output_dir")
    _int(cfg["seed"], f"{source}: seed")
    for key in ("sample_hz", "training_hz"):
        if _num(cfg[key], f"{source}: {key}") <= 0.0:
            raise ValueError(f"{source}: {key} 必须为正")
    if float(cfg["training_hz"]) > float(cfg["sample_hz"]):
        raise ValueError(f"{source}: training_hz 不能高于 sample_hz")
    prefixes = cfg["prefixes_s"]
    if not isinstance(prefixes, (list, tuple)) or not prefixes:
        raise ValueError(f"{source}: prefixes_s 必须是非空列表")
    for i, v in enumerate(prefixes):
        if _num(v, f"{source}: prefixes_s[{i}]") <= 0.0:
            raise ValueError(f"{source}: prefixes_s 必须为正")
    if list(prefixes) != sorted(prefixes):
        raise ValueError(f"{source}: prefixes_s 必须递增")
    # 四段 link 的 5 种温度情景：每个映射 = 区域名 -> [lo, hi]。
    # 区域名与 region_order 的一致性在采样器里跨文件校验（此处只看结构）。
    cases = cfg["region_temps_c"]
    if not isinstance(cases, (list, tuple)) or not cases:
        raise ValueError(f"{source}: region_temps_c 必须是非空列表（每种情景一个映射）")
    for i, case in enumerate(cases):
        if not isinstance(case, dict) or not case:
            raise ValueError(f"{source}: region_temps_c[{i}] 必须是非空映射（区域名 -> [lo, hi]）")
        for name, span in case.items():
            _str(name, f"{source}: region_temps_c[{i}] 的键")
            _range(span, f"{source}: region_temps_c[{i}].{name}", positive=False)

    finger = cfg["fingertip"]
    _check_keys(
        finger,
        required=frozenset(
            {
                "history_mode_c",
                "history_probability",
                "pair_delta_c",
                "physical_range_c",
            }
        ),
        optional=frozenset(),
        where=f"{source}: fingertip",
    )
    _range(finger["history_mode_c"], f"{source}: fingertip.history_mode_c", positive=False)
    _range(finger["pair_delta_c"], f"{source}: fingertip.pair_delta_c", positive=False)
    _range(finger["physical_range_c"], f"{source}: fingertip.physical_range_c", positive=False)
    p = _num(finger["history_probability"], f"{source}: fingertip.history_probability")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"{source}: fingertip.history_probability 必须在 [0, 1]")

    levels = cfg["force_levels"]
    if not isinstance(levels, (list, tuple)) or not levels:
        raise ValueError(f"{source}: force_levels 必须是非空列表")
    for i, v in enumerate(levels):
        _num(v, f"{source}: force_levels[{i}]")

    quality = cfg["contact_quality_probabilities"]
    _check_keys(
        quality,
        required=frozenset({"stable", "brief", "failed"}),
        optional=frozenset(),
        where=f"{source}: contact_quality_probabilities",
    )
    qtotal = sum(_num(v, f"{source}: contact_quality_probabilities.{k}") for k, v in quality.items())
    if abs(qtotal - 1.0) > 1e-6:
        raise ValueError(f"{source}: contact_quality_probabilities 之和必须为 1，实际 {qtotal}")

    sensor = cfg["sensor"]
    _check_keys(
        sensor,
        required=frozenset({"rc_s", "noise_c", "bias_c", "quantization_c", "shift_test"}),
        optional=frozenset(),
        where=f"{source}: sensor",
    )
    # 三个乘性量（RC / 噪声 / 量化）用 multiplier_range 缩放名义值；
    # 偏置是加性 °C，用 absolute_range。
    for key in ("rc_s", "noise_c", "quantization_c"):
        entry = sensor[key]
        _check_keys(
            entry,
            required=frozenset({"nominal", "multiplier_range"}),
            optional=frozenset(),
            where=f"{source}: sensor.{key}",
        )
        _num(entry["nominal"], f"{source}: sensor.{key}.nominal")
        lo, hi = _num_list(entry["multiplier_range"], f"{source}: sensor.{key}.multiplier_range", 2)
        if lo <= 0.0 or lo > hi:
            raise ValueError(f"{source}: sensor.{key}.multiplier_range 必须是正区间 [lo, hi]")
    bias = sensor["bias_c"]
    _check_keys(
        bias,
        required=frozenset({"nominal", "absolute_range"}),
        optional=frozenset(),
        where=f"{source}: sensor.bias_c",
    )
    _num(bias["nominal"], f"{source}: sensor.bias_c.nominal")
    _range(bias["absolute_range"], f"{source}: sensor.bias_c.absolute_range", positive=False)
    shift = sensor["shift_test"]
    _check_keys(
        shift,
        required=frozenset(
            {
                "rc_multiplier_range",
                "noise_multiplier_range",
                "quantization_multiplier_range",
                "bias_absolute_range",
            }
        ),
        optional=frozenset(),
        where=f"{source}: sensor.shift_test",
    )
    for key in ("rc_multiplier_range", "noise_multiplier_range", "quantization_multiplier_range"):
        lo, hi = _num_list(shift[key], f"{source}: sensor.shift_test.{key}", 2)
        if lo <= 0.0 or lo > hi:
            raise ValueError(f"{source}: sensor.shift_test.{key} 必须是正区间")
    _range(shift["bias_absolute_range"], f"{source}: sensor.shift_test.bias_absolute_range", positive=False)

    splits = cfg["splits"]
    if not isinstance(splits, dict) or not splits:
        raise ValueError(f"{source}: splits 必须是非空映射")
    for name, count in splits.items():
        _str(name, f"{source}: splits 的键")
        _int(count, f"{source}: splits.{name}", minimum=0)
    return cfg


__all__ = [
    "REPO_ROOT",
    "TaskConfig",
    "load_dataset_config",
    "load_object_config",
    "load_sensor_config",
    "load_task_config",
]

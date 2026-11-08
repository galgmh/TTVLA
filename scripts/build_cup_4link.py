#!/usr/bin/env python3
"""生成四段式杯子资产（ThermalContact-v1）。

杯子被拆成四个 link —— 杯身下 / 中 / 上三分之一，加手柄 —— 因为引擎的
温度场给每个 link 一个集总温度（``link_temps``）。指尖接触传热核读取的
正是所接触 link 的那个标量，因此"指尖读到多少"直接由"碰到哪一段"决定，
不需要物体侧的 TemperatureGrid。

输出目录（默认 ``assets/cup_4link``）：

* ``meshes/cup.obj`` —— 杯身 + 手柄的合并体，只作为*测量参考*：AABB、体积、
  总质量由它决定，拆分不改变这些量。
* ``meshes/lower.obj`` —— 下三分之一：杯底 + 到第一个切面的杯壁，封闭实体
  （``body_mesh``，高度取 H/3）。
* ``meshes/middle.obj`` / ``meshes/upper.obj`` —— 中 / 上三分之一的环形管
  （``tube_mesh``），上段顶到杯口。
* ``meshes/handle.obj`` —— 完整可见手柄（190° 弧），两端埋进杯壁。
* ``meshes/handle_collision.obj`` —— 同一圆环的外侧 130°，作为手柄 link 的
  *碰撞*网格。所有顶点都在杯体外表面之外 ≥ 8 mm，因此不可能与杯身段重叠。
  这很重要：引擎的相邻 link 过滤只沿父子链走（collider.py），而
  ``upper`` 与 ``handle`` 是兄弟关系，否则要靠构建时的中性位形体素检查。
  用这个代理网格后，这一对根本不会被生成。
* ``cup_4link.urdf`` —— 四个 link、三个固定关节，所有 link 坐标系都取
  “作者坐标系”（原点 = 杯底中心，+z 沿杯轴，手柄在 -x）：

      lower --> middle --> upper
                   |
                handle

  总质量由 ``--mass``（0.22 kg）按各段体积分配；每段惯量是 trimesh 绕自身
  质心的质量属性。构建成 URDF 之后，link 质量就由这些 <inertial> 决定。

作者坐标系是任务配置里 ``regions[*].pos`` 所表达的坐标系；场景按
``base_euler_deg = [0, 0, 90]`` 把它与世界对齐，于是作者的 -x 手柄方向
变成世界 -y。

    python scripts/build_cup_4link.py
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

ASSET_VERSION = "cup4link-v1"


def _ring(radius: float, z: float, n: int) -> np.ndarray:
    """z 平面上的等分圆环顶点。"""
    a = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(a), radius * np.sin(a), np.full(n, z, dtype=np.float64)])


def _bridge(off_a: int, off_b: int, n: int) -> list[list[int]]:
    """两个等分圆环之间的四边形带，每个四边形拆成两个三角形。"""
    faces: list[list[int]] = []
    for k in range(n):
        k2 = (k + 1) % n
        a0, a1 = off_a + k, off_a + k2
        b0, b1 = off_b + k, off_b + k2
        faces.append([a0, a1, b1])
        faces.append([a0, b1, b0])
    return faces


def _fan(apex: int, off: int, n: int, *, reverse: bool) -> list[list[int]]:
    """以轴心点为顶点的三角扇。"""
    faces: list[list[int]] = []
    for k in range(n):
        k2 = (k + 1) % n
        faces.append([apex, off + k2, off + k] if reverse else [apex, off + k, off + k2])
    return faces


def body_mesh(radius: float, height: float, wall: float, bottom: float, n: int):
    """开口朝上的中空杯体，作为封闭旋转面。

    轮廓 (r, z) 闭合：(0,0) -> (R,0) -> (R,H) -> (R-w,H) -> (R-w,b) ->
    (0,b) -> (0,0)。两个轴心点变成三角扇顶点，四条径向边变成杯底 / 外壁 /
    杯口 / 内壁四条带。

    令 ``height = H/3`` 时它就是杯子的*下三分之一*：杯底加上到第一个切面
    的杯壁。
    """
    verts = [
        np.array([0.0, 0.0, 0.0]),  # 0：外侧杯底中心
        _ring(radius, 0.0, n),  # 1：外侧杯底边缘
        _ring(radius, height, n),  # 1+n：外侧顶部（杯口外沿）
        _ring(radius - wall, height, n),  # 1+2n：内侧顶部（杯口内沿）
        _ring(radius - wall, bottom, n),  # 1+3n：内侧杯底
        np.array([0.0, 0.0, bottom]),  # 1+4n：内侧杯底中心
    ]
    v = np.vstack(verts)
    ob, ot, it, ib = 1, 1 + n, 1 + 2 * n, 1 + 3 * n
    faces: list[list[int]] = []
    faces += _fan(0, ob, n, reverse=True)  # 杯底外表面，朝 -z
    faces += _bridge(ob, ot, n)  # 外壁
    faces += _bridge(ot, it, n)  # 杯口
    faces += _bridge(it, ib, n)  # 内壁
    faces += _fan(1 + 4 * n, ib, n, reverse=False)  # 杯底内表面，朝 +z
    return v, np.asarray(faces, dtype=np.int64)


def tube_mesh(radius: float, wall: float, z0: float, z1: float, n: int):
    """封闭环形管：两个 z 平面之间的杯壁，两端加盖。

    四个圆环 —— 外 z0、外 z1、内 z1、内 z0 —— 加四条带（外壁、上环面、
    内壁、下环面）。杯身的中 / 上三分之一正是这个形状。
    """
    ri = radius - wall
    verts = np.vstack(
        [
            _ring(radius, z0, n),  # 0：外下
            _ring(radius, z1, n),  # n：外上
            _ring(ri, z1, n),  # 2n：内上
            _ring(ri, z0, n),  # 3n：内下
        ]
    )
    faces: list[list[int]] = []
    faces += _bridge(0, n, n)  # 外壁
    faces += _bridge(n, 2 * n, n)  # 上环面
    faces += _bridge(2 * n, 3 * n, n)  # 内壁
    faces += _bridge(3 * n, 0, n)  # 下环面
    return verts, np.asarray(faces, dtype=np.int64)


def handle_mesh(
    major: float,
    tube_r: float,
    tube_y: float,
    xc: float,
    zc: float,
    arc_deg: float,
    m: int,
    k: int,
):
    """沿 x-z 平面内一段圆弧扫掠的提手，两端封盖。

    截面是椭圆：沿径向 ``er`` 的半轴 ``tube_r``（提手的前后厚度）、沿作者
    y 轴的半轴 ``tube_y``（提手宽度）。圆管会让平行夹爪绕自身接触轴滚动，
    杯子会在夹爪下摆动约 80° 并把杯身拖到桌面上；深度大于宽度的椭圆截面
    无法旋转 —— 必须先张开夹爪。

    ``arc_deg < 190`` 得到更短的弧，端部离杯轴更远；默认 130° 是无卡箍的
    碰撞代理（见模块 docstring）。
    """
    half = 0.5 * np.deg2rad(arc_deg)
    # 弧以 -x 方向的竖条（theta = 180°）为中心，朝 +x（杯身内部）张开：
    # theta 从 180-half 到 180+half，因此两端分别落在杯壁内（完整弧）
    # 或远离杯壁（碰撞代理）。
    th0 = np.pi - half
    th1 = np.pi + half
    th = np.linspace(th0, th1, m + 1)
    ph = np.linspace(0.0, 2.0 * np.pi, k, endpoint=False)
    centre = np.array([xc, 0.0, zc])
    ey = np.array([0.0, 1.0, 0.0])
    verts = []
    for t in th:
        er = np.array([np.cos(t), 0.0, np.sin(t)])
        c = centre + major * er
        for p in ph:
            verts.append(c + tube_r * np.cos(p) * er + tube_y * np.sin(p) * ey)
    faces: list[list[int]] = []
    for i in range(m):
        for j in range(k):
            j2 = (j + 1) % k
            a0, a1 = i * k + j, i * k + j2
            b0, b1 = (i + 1) * k + j, (i + 1) * k + j2
            faces.append([a0, a1, b1])
            faces.append([a0, b1, b0])
    # 两个端盖：从两端圆环中心做三角扇。扫掠切向为
    # t = (-sin, 0, cos)；起始端盖朝 -t，终止端盖朝 +t。
    base = (m + 1) * k
    verts.append(centre + major * np.array([np.cos(th0), 0.0, np.sin(th0)]))
    verts.append(centre + major * np.array([np.cos(th1), 0.0, np.sin(th1)]))
    for j in range(k):
        j2 = (j + 1) % k
        faces.append([base, j2, j])  # 起始端盖
        faces.append([base + 1, m * k + j, m * k + j2])  # 终止端盖
    return np.asarray(verts), np.asarray(faces, dtype=np.int64)


def _repair(verts: np.ndarray, faces: np.ndarray):
    """构建 trimesh、焊接重复顶点并把法线朝外。"""
    import trimesh

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    trimesh.repair.fix_normals(mesh)
    return mesh


def _concat(a, b):
    """拼接两个网格（trimesh 懒导入，与 :func:`_repair` 一致）。"""
    import trimesh

    return trimesh.util.concatenate([a, b])


def _check(name: str, mesh) -> None:
    """封闭性与体积检查：引擎需要闭合实体。"""
    if not mesh.is_watertight:
        raise SystemExit(f"{name}：网格不封闭，引擎需要闭合实体")
    if mesh.volume <= 0.0:
        raise SystemExit(f"{name}：体积非正（{mesh.volume:.3e} m^3）")


def _print_part(name: str, mesh) -> None:
    lo, hi = mesh.bounds
    print(
        f"  {name:<22} 面 {len(mesh.faces):<6} 封闭 {str(mesh.is_watertight):<5} "
        f"体积 {mesh.volume * 1e6:8.3f} cm^3  z [{lo[2]:.4f}, {hi[2]:.4f}]"
    )


def _inertial_xml(mesh, mass: float) -> str:
    """URDF <inertial>：trimesh 绕该段自身质心的质量属性。"""
    # 这个版本的 trimesh 把 ``mass_properties`` 暴露为按 ``mesh.density``
    # 缓存的*属性*（默认单位密度），不是可调用方法；把密度设成
    # 质量/体积，返回的质量就正好等于 link 质量。
    mesh.density = float(mass) / float(mesh.volume)
    props = mesh.mass_properties
    if not np.isclose(float(props.mass), float(mass), rtol=1e-9, atol=0.0):
        raise SystemExit(
            f"mass_properties 返回 {float(props.mass):.9e} kg，期望 {float(mass):.9e} kg"
        )
    com = np.asarray(props.center_mass, dtype=np.float64).reshape(3)
    inertia = np.asarray(props.inertia, dtype=np.float64).reshape(3, 3)
    eig = np.linalg.eigvalsh(inertia)
    if eig.min() <= 0.0:
        raise SystemExit(
            f"绕质心的惯量不正定（最小特征值 {eig.min():.3e}）；"
            "薄壳可能需要简化的碰撞/惯量形状"
        )

    def f(v: float) -> str:
        return f"{float(v):.9g}"

    return (
        "    <inertial>\n"
        f'      <origin xyz="{f(com[0])} {f(com[1])} {f(com[2])}" rpy="0 0 0"/>\n'
        f'      <mass value="{f(mass)}"/>\n'
        f'      <inertia ixx="{f(inertia[0, 0])}" ixy="{f(inertia[0, 1])}" ixz="{f(inertia[0, 2])}"'
        f' iyy="{f(inertia[1, 1])}" iyz="{f(inertia[1, 2])}" izz="{f(inertia[2, 2])}"/>\n'
        "    </inertial>\n"
    )


def _link_xml(name: str, visual: str, collision: str, inertial: str) -> str:
    """单个 link：视觉网格 + 碰撞网格（文件名相对 URDF 所在目录）。"""
    return (
        f'  <link name="{name}">\n'
        f"{inertial}"
        "    <visual>\n"
        '      <origin xyz="0 0 0" rpy="0 0 0"/>\n'
        f'      <geometry><mesh filename="{visual}"/></geometry>\n'
        "    </visual>\n"
        "    <collision>\n"
        '      <origin xyz="0 0 0" rpy="0 0 0"/>\n'
        f'      <geometry><mesh filename="{collision}"/></geometry>\n'
        "    </collision>\n"
        "  </link>\n"
    )


def _joint_xml(name: str, parent: str, child: str) -> str:
    """固定关节；所有 link 坐标系同为作者坐标系，故 origin 恒为单位。"""
    return (
        f'  <joint name="{name}" type="fixed">\n'
        f'    <parent link="{parent}"/>\n'
        f'    <child link="{child}"/>\n'
        '    <origin xyz="0 0 0" rpy="0 0 0"/>\n'
        "  </joint>\n"
    )


def _write_urdf(path: Path, parts: list[tuple[str, str, str, object, float]]) -> None:
    """写四 link URDF，并重新解析以检查每个引用。

    ``parts`` 每行是（link 名、视觉 OBJ 相对路径、碰撞 OBJ 相对路径、
    惯量网格、质量 kg）。所有 link 坐标系都是作者坐标系，故关节 origin
    全为单位变换。
    """
    xml = '<?xml version="1.0"?>\n<robot name="cup_4link">\n'
    for name, visual, collision, mesh, mass in parts:
        xml += _link_xml(name, visual, collision, _inertial_xml(mesh, mass))
    xml += _joint_xml("lower_middle", "lower", "middle")
    xml += _joint_xml("middle_upper", "middle", "upper")
    xml += _joint_xml("middle_handle", "middle", "handle")
    xml += "</robot>\n"
    path.write_text(xml, encoding="utf-8")

    root = ET.fromstring(path.read_text(encoding="utf-8"))
    links = [e.get("name") for e in root.findall("link")]
    joints = [
        (e.get("name"), e.get("type"), e.find("parent").get("link"), e.find("child").get("link"))
        for e in root.findall("joint")
    ]
    for link in root.findall("link"):
        for tag in ("visual", "collision"):
            geom = link.find(f"{tag}/geometry/mesh")
            if geom is None:
                raise SystemExit(f"{path}：{link.get('name')} 缺少 {tag} 网格")
            if not (path.parent / geom.get("filename")).is_file():
                raise SystemExit(
                    f"{path}：{link.get('name')} 的 {tag} 网格不存在：{geom.get('filename')}"
                )
    print(f"写出 {path}")
    print(f"  links  ：{links}")
    print(f"  joints ：{joints}")


def _manifest(
    path: Path,
    *,
    asset_version: str,
    urdf_name: str,
    z_spans: dict[str, list[float]],
    link_names: list[str],
    base_euler_deg: list[float],
    mass_kg: float,
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
) -> None:
    """写资产清单：区域顺序 / link 名 / z 分界 / 质量 / AABB。

    清单是"区域 → link"的唯一真源：采样器只认 ``region_order``，运行时只
    按 ``regions[name].link`` 解析 link 索引，代码里不再写死任何 link 名。
    """
    doc = {
        "asset_version": asset_version,
        "urdf_file": urdf_name,
        # 区域顺序 = 温度向量 privileged/link_temp 的列顺序。
        "region_order": ["lower", "middle", "upper", "handle"],
        "regions": {
            name: {"link": name, "z_span_m": [float(z_spans[name][0]), float(z_spans[name][1])]}
            for name in ("lower", "middle", "upper", "handle")
        },
        "link_names": list(link_names),
        # 场景用这个欧拉角把作者坐标系对齐到世界：-x 手柄方向 -> 世界 -y。
        "base_euler_deg": [float(v) for v in base_euler_deg],
        "mass_kg": float(mass_kg),
        "aabb_min_m": [float(v) for v in aabb_min],
        "aabb_max_m": [float(v) for v in aabb_max],
    }
    path.write_text(
        yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8",
    )
    print(f"写出 {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="assets/cup_4link", help="资产目录")
    ap.add_argument("--radius", type=float, default=0.030, help="杯体外半径 (m)")
    ap.add_argument("--height", type=float, default=0.110, help="杯体高度 (m)")
    ap.add_argument("--wall", type=float, default=0.005, help="壁厚 (m)")
    ap.add_argument("--bottom", type=float, default=0.007, help="杯底厚度 (m)")
    ap.add_argument("--handle-major", type=float, default=0.027, help="手柄圆环半径 (m)")
    ap.add_argument("--handle-tube", type=float, default=0.008, help="手柄半厚（径向，m）")
    ap.add_argument("--handle-tube-y", type=float, default=0.005, help="手柄半宽（作者 y，m）")
    ap.add_argument("--handle-z", type=float, default=0.056, help="手柄中心高度 (m)")
    ap.add_argument("--handle-arc-deg", type=float, default=190.0)
    ap.add_argument(
        "--handle-collision-arc-deg",
        type=float,
        default=130.0,
        help="手柄碰撞代理的弧角；两端必须离开杯壁",
    )
    ap.add_argument("--body-segments", type=int, default=48)
    ap.add_argument("--handle-segments", type=int, default=24)
    ap.add_argument("--handle-sides", type=int, default=10)
    ap.add_argument("--mass", type=float, default=0.22, help="URDF 总质量 (kg)")
    ap.add_argument("--asset-version", default=ASSET_VERSION)
    args = ap.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    mesh_dir = out_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    # ---- 合并体（杯身 + 手柄）：测量参考，拆分不改变它 ----------------------
    body = _repair(*body_mesh(args.radius, args.height, args.wall, args.bottom, args.body_segments))
    handle = _repair(
        *handle_mesh(
            args.handle_major,
            args.handle_tube,
            args.handle_tube_y,
            -args.radius,  # 圆心落在 -x 杯壁上，外侧竖条才能避开杯体
            args.handle_z,
            args.handle_arc_deg,
            args.handle_segments,
            args.handle_sides,
        )
    )
    _check("body", body)
    _check("handle", handle)
    cup = _concat(body, handle)
    cup.export(mesh_dir / "cup.obj")

    lo, hi = cup.bounds
    print(f"写出 {mesh_dir / 'cup.obj'}")
    print(f"  面数       ：{len(cup.faces)}（杯身 {len(body.faces)} + 手柄 {len(handle.faces)}）")
    print(f"  封闭       ：body={body.is_watertight} handle={handle.is_watertight} all={cup.is_watertight}")
    print(
        f"  体积       ：{body.volume * 1e6:.1f} + {handle.volume * 1e6:.1f} "
        f"= {(body.volume + handle.volume) * 1e6:.1f} cm^3"
    )
    print(f"  AABB 下界  ：{np.round(lo, 4)}")
    print(f"  AABB 上界  ：{np.round(hi, 4)}")

    # ---- 拆分：下 / 中 / 上三分之一 + 手柄 ----------------------------------
    h1 = args.height / 3.0
    h2 = 2.0 * args.height / 3.0
    lower = _repair(*body_mesh(args.radius, h1, args.wall, args.bottom, args.body_segments))
    middle = _repair(*tube_mesh(args.radius, args.wall, h1, h2, args.body_segments))
    upper = _repair(*tube_mesh(args.radius, args.wall, h2, args.height, args.body_segments))
    handle_col = _repair(
        *handle_mesh(
            args.handle_major,
            args.handle_tube,
            args.handle_tube_y,
            -args.radius,
            args.handle_z,
            args.handle_collision_arc_deg,
            args.handle_segments,
            args.handle_sides,
        )
    )
    for name, mesh in (
        ("lower", lower),
        ("middle", middle),
        ("upper", upper),
        ("handle", handle),
        ("handle_collision", handle_col),
    ):
        _check(name, mesh)

    print(f"  切面       ：z = {h1:.4f} / {h2:.4f} m（{args.height:.4f} m 的三等分）")
    _print_part("lower", lower)
    _print_part("middle", middle)
    _print_part("upper", upper)
    _print_part("handle", handle)
    _print_part("handle_collision", handle_col)

    # 四段应恰好分割合并体：体积与 AABB 必须一致。
    split_volume = lower.volume + middle.volume + upper.volume + handle.volume
    if not np.isclose(split_volume, cup.volume, rtol=1e-6, atol=1e-12):
        raise SystemExit(f"分段体积 {split_volume:.9e} != 合并体 {cup.volume:.9e} m^3")
    part_lo = np.min([m.bounds[0] for m in (lower, middle, upper, handle)], axis=0)
    part_hi = np.max([m.bounds[1] for m in (lower, middle, upper, handle)], axis=0)
    if not (np.allclose(part_lo, lo) and np.allclose(part_hi, hi)):
        raise SystemExit(f"分段 AABB {part_lo}/{part_hi} != 合并体 {lo}/{hi}")
    print(
        f"  分割校验   ：体积 {split_volume * 1e6:.3f} cm^3 == 合并体，"
        f"AABB {np.round(part_lo, 4)} .. {np.round(part_hi, 4)}"
    )

    # 手柄碰撞代理必须离开杯体外壁，这样 upper<->handle 兄弟对根本不会
    # 进入碰撞器。
    r_min = float(np.hypot(handle_col.vertices[:, 0], handle_col.vertices[:, 1]).min())
    clearance = r_min - args.radius
    if clearance <= 0.005:
        raise SystemExit(
            f"手柄碰撞代理伸到 r = {r_min:.4f} m（间隙 {clearance * 1e3:.1f} mm）；"
            "请减小 --handle-collision-arc-deg"
        )
    print(f"  手柄碰撞   ：最小半径 {r_min * 1e3:.1f} mm，与杯壁间隙 {clearance * 1e3:.1f} mm")

    # ---- 写四段网格 + URDF + 清单 -------------------------------------------
    names = {
        "lower": "meshes/lower.obj",
        "middle": "meshes/middle.obj",
        "upper": "meshes/upper.obj",
        "handle": "meshes/handle.obj",
        "handle_collision": "meshes/handle_collision.obj",
    }
    for mesh, rel in (
        (lower, names["lower"]),
        (middle, names["middle"]),
        (upper, names["upper"]),
        (handle, names["handle"]),
        (handle_col, names["handle_collision"]),
    ):
        mesh.export(out_dir / rel)

    total_volume = float(split_volume)
    volumes = [lower.volume, middle.volume, upper.volume, handle.volume]
    masses = [args.mass * v / total_volume for v in volumes]
    _write_urdf(
        out_dir / "cup_4link.urdf",
        [
            ("lower", names["lower"], names["lower"], lower, masses[0]),
            ("middle", names["middle"], names["middle"], middle, masses[1]),
            ("upper", names["upper"], names["upper"], upper, masses[2]),
            # 视觉 = 完整提手；碰撞 = 离开杯壁的代理。
            ("handle", names["handle"], names["handle_collision"], handle, masses[3]),
        ],
    )
    print(
        "  质量       ："
        + "  ".join(f"{n} {m * 1e3:.1f} g" for n, m in zip(("lower", "middle", "upper", "handle"), masses))
        + f"  （合计 {sum(masses) * 1e3:.1f} g）"
    )
    print(
        "  手柄竖条   ："
        f"x={-args.radius - args.handle_major:+.4f} z={args.handle_z:.4f} "
        "（任务配置里 handle 区域的 pos 用它）"
    )

    _manifest(
        out_dir / "manifest.yaml",
        asset_version=args.asset_version,
        urdf_name="cup_4link.urdf",
        z_spans={
            "lower": [0.0, h1],
            "middle": [h1, h2],
            "upper": [h2, args.height],
            # 手柄的热区由 link 本身界定，z 范围取网格实测（含弧形两端）。
            "handle": [float(handle.bounds[0][2]), float(handle.bounds[1][2])],
        },
        link_names=["lower", "middle", "upper", "handle"],
        base_euler_deg=[0.0, 0.0, 90.0],
        mass_kg=float(sum(masses)),
        aabb_min=lo,
        aabb_max=hi,
    )


if __name__ == "__main__":
    main()

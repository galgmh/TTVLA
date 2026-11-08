"""在平行夹爪指尖上程序化推导触觉探针布局。

Panda 的 URDF 把指尖几何写成碰撞**网格**（腕部 link 是 box）。与其把垫面
偏移写死，不如运行时解析 URDF，在"朝向另一根手指"的那个面上合成探针网格。
探针坐标表达在手指 link 坐标系里——这正是 Genesis 基于探针的触觉传感器
所期望的。

整指 AABB 是错误的垫面模型：Panda 指尖网格只在靠近指尖处（link 坐标
z ≈ 0.036..0.054）有一块凸起的平面垫，根部与腰部凹进去 2.5..15 mm。放在
AABB 面上的探针会悬在真实表面之前，夹爪提前闭合且深度读数虚高。因此对
网格碰撞我们取网格**顶点**，选出垫面那一层（到最内侧表面的距离在
``pad_tol_m`` 以内的点，``d = v @ inner``），再把每个探针吸附到它自己的
切向邻域内的局部表面高度。``pad_tol_m`` 必须小于凹陷深度，否则这一层会
和手指其他部分连成一片：在 Panda 上 0.5..2 mm 的任何厚度都得到同一张垫面
（28 个顶点，17.4 x 17.5 mm），默认 1.5 mm 落在稳定窗口中间。

闭合轴取自共享父坐标系（手）里两个碰撞形状中心之差——不能用关节原点
（Panda 这类 mimic 夹爪的两指关节原点重合，都是 ``xyz="0 0 0.0584"``），
也不能用垫面中心（夹爪闭合时两者只差 0.2 mm，是噪声）。没有碰撞几何时
回落到关节原点。

纯 XML + numpy：本模块不导入 Genesis，布局可以独立检查和单元测试。
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class FingerLayout:
    """单个指尖 link 的探针网格（link 局部坐标系）。"""

    link_name: str
    probe_local_pos: np.ndarray  # (P, 3) float32
    probe_local_normal: np.ndarray  # (P, 3) float32，单位向量
    pad_box_origin: np.ndarray  # (3,) 垫面中心（link 系）
    pad_box_size: np.ndarray  # (3,) 垫面范围；沿内侧轴为 0
    inner_dir: np.ndarray  # (3,) 指向另一根手指的单位向量

    @property
    def pad_face_area_m2(self) -> float:
        """垫面面积（m²）：非零两个方向尺度的乘积。

        ``pad_box_size`` 沿内侧（接触法向）轴为 0，所以这就是抓取面自身的
        面积——Panda 上 305.949 mm²。pad_face 面积策略把它当作手指所有
        接触点分摊的那一块接触面积。
        """
        size = np.abs(np.asarray(self.pad_box_size, dtype=np.float64))
        return float(np.prod(size[size > 0.0]))


def _parse_vec(text: str | None, default: tuple[float, ...]) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(v) for v in text.split()], dtype=np.float64)


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """URDF 固定轴约定：R = Rz(yaw) @ Ry(pitch) @ Rx(roll)。"""
    r, p, y = (float(v) for v in rpy)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


@dataclass
class _Box:
    origin: np.ndarray  # (3,)
    size: np.ndarray  # (3,)
    rpy: np.ndarray  # (3,)


def parse_link_boxes(urdf_path: str | Path, links: set[str] | None = None) -> dict[str, list[_Box]]:
    """link 名 -> 碰撞形状列表（link 系下的 origin/size/rpy）。

    ``<box>`` 碰撞直接读取；``<mesh>`` 碰撞（.obj）退化为 link 系下的轴对齐
    包围盒（已施加碰撞 origin）。``derive_finger_layout`` 优先用
    ``parse_link_meshes`` 的原始顶点，这个 AABB 视图只是粗兜底。``links``
    可限制解析范围（大 URDF）。
    """
    root = ET.parse(urdf_path).getroot()
    out: dict[str, list[_Box]] = {}
    for link in root.findall("link"):
        name = link.get("name")
        if links is not None and name not in links:
            continue
        boxes: list[_Box] = []
        for coll in link.findall("collision"):
            origin_el = coll.find("origin")
            xyz = _parse_vec(origin_el.get("xyz") if origin_el is not None else None, (0.0, 0.0, 0.0))
            rpy = _parse_vec(origin_el.get("rpy") if origin_el is not None else None, (0.0, 0.0, 0.0))
            box = coll.find("geometry/box")
            if box is not None:
                boxes.append(_Box(origin=xyz, size=_parse_vec(box.get("size"), (0.0, 0.0, 0.0)), rpy=rpy))
                continue
            mesh = coll.find("geometry/mesh")
            if mesh is not None:
                verts = _mesh_vertices(urdf_path, mesh, xyz, rpy)
                if verts is not None:
                    lo, hi = verts.min(axis=0), verts.max(axis=0)
                    boxes.append(_Box(origin=(lo + hi) / 2.0, size=hi - lo, rpy=np.zeros(3)))
                continue
        if boxes:
            out[name] = boxes
    return out


def _resolve_mesh_path(urdf_path: str | Path, filename: str) -> Path | None:
    """把 URDF 的网格引用（``package://`` 或相对路径）解析成文件。"""
    if not filename:
        return None
    if filename.startswith("package://"):
        rel = Path(filename[len("package://") :])
        base = Path(urdf_path).resolve().parent
        for cand in (base / rel, *(a / rel for a in base.parents)):
            if cand.is_file():
                return cand
        return None
    cand = Path(filename)
    if not cand.is_absolute():
        cand = Path(urdf_path).resolve().parent / cand
    return cand if cand.is_file() else None


def _obj_vertices(path: Path, scale: np.ndarray) -> np.ndarray:
    """Wavefront .obj 的 (N,3) 顶点（只读 ``v`` 行；纯 numpy）。"""
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        parts = line.split()
        if parts and parts[0] == "v" and len(parts) >= 4:
            rows.append([float(x) for x in parts[1:4]])
    if not rows:
        raise ValueError(f"网格 {path} 没有顶点")
    return np.asarray(rows, dtype=np.float64) * scale


def _mesh_vertices(urdf_path: str | Path, mesh_el, xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray | None:
    """link 系下的碰撞网格顶点（已施加碰撞 origin）；不可用返回 None。"""
    path = _resolve_mesh_path(urdf_path, mesh_el.get("filename") or "")
    if path is None or path.suffix.lower() != ".obj":
        return None
    scale = _parse_vec(mesh_el.get("scale"), (1.0, 1.0, 1.0))
    return _obj_vertices(path, scale) @ rpy_to_matrix(rpy).T + xyz


def parse_link_meshes(urdf_path: str | Path, links: set[str] | None = None) -> dict[str, np.ndarray]:
    """link 名 -> 拼接后的碰撞网格顶点（.obj，link 系）。

    这些是垫面分层所需的原始点；``parse_link_boxes`` 保留较粗的 AABB 视图，
    给只需要尺度的调用者。
    """
    root = ET.parse(urdf_path).getroot()
    out: dict[str, np.ndarray] = {}
    for link in root.findall("link"):
        name = link.get("name")
        if links is not None and name not in links:
            continue
        chunks = []
        for coll in link.findall("collision"):
            origin_el = coll.find("origin")
            xyz = _parse_vec(origin_el.get("xyz") if origin_el is not None else None, (0.0, 0.0, 0.0))
            rpy = _parse_vec(origin_el.get("rpy") if origin_el is not None else None, (0.0, 0.0, 0.0))
            mesh = coll.find("geometry/mesh")
            if mesh is None:
                continue
            verts = _mesh_vertices(urdf_path, mesh, xyz, rpy)
            if verts is not None:
                chunks.append(verts)
        if chunks:
            out[name] = np.concatenate(chunks, axis=0)
    return out


@dataclass
class _Joint:
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray


def parse_child_joints(urdf_path: str | Path) -> dict[str, _Joint]:
    """子 link 名 -> 它的父关节（origin 在父坐标系里）。"""
    root = ET.parse(urdf_path).getroot()
    out: dict[str, _Joint] = {}
    for joint in root.findall("joint"):
        child_el = joint.find("child")
        parent_el = joint.find("parent")
        if child_el is None or parent_el is None:
            continue
        origin_el = joint.find("origin")
        out[child_el.get("link")] = _Joint(
            parent=parent_el.get("link"),
            child=child_el.get("link"),
            xyz=_parse_vec(origin_el.get("xyz") if origin_el is not None else None, (0.0, 0.0, 0.0)),
            rpy=_parse_vec(origin_el.get("rpy") if origin_el is not None else None, (0.0, 0.0, 0.0)),
        )
    return out


def _union_box(boxes: list[_Box]) -> tuple[np.ndarray, np.ndarray]:
    """若干可能旋转过的 box 在 link 系下的轴对齐并集。"""
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for box in boxes:
        rot = rpy_to_matrix(box.rpy)
        half = box.size / 2.0
        corners = np.array(
            [[sx * half[0], sy * half[1], sz * half[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        )
        world = corners @ rot.T + box.origin
        lo = np.minimum(lo, world.min(axis=0))
        hi = np.maximum(hi, world.max(axis=0))
    return (lo + hi) / 2.0, hi - lo


def _snap_to_axis(vec: np.ndarray, *, tol: float = 0.9) -> np.ndarray:
    """把近似轴向的向量吸附到坐标轴。"""
    axis = int(np.argmax(np.abs(vec)))
    if abs(vec[axis]) < tol:
        raise ValueError(f"内侧方向 {vec} 不是轴向的；请在 link 系里传 inner_dir_override")
    out = np.zeros(3)
    out[axis] = math.copysign(1.0, vec[axis])
    return out


@dataclass
class _PadFace:
    """一根手指的抓取面（link 系）。

    ``center`` / ``size`` 描述垫面平面（``size`` 沿内侧轴为 0）；``slab``
    保存垫面自己的网格顶点（逐探针的表面查找只在这些点里搜，因此探针不会
    吸附到凹陷的手指主体上）；box 碰撞时为 None，面是平的。
    """

    center: np.ndarray
    size: np.ndarray
    axis: int
    d_max: float
    slab: np.ndarray | None


def _closure_direction(
    center_a: np.ndarray, center_b: np.ndarray, joint_a: _Joint, joint_b: _Joint
) -> tuple[np.ndarray | None, str]:
    """父坐标系下 a -> b 的单位向量，以及它的来源。

    ``center_*`` 必须是碰撞形状中心（相距足够远），绝不能用垫面中心（夹爪
    闭合时几乎共面）。
    """
    d = np.asarray(center_b, dtype=np.float64) - np.asarray(center_a, dtype=np.float64)
    if np.linalg.norm(d) > 1e-6:
        return d / np.linalg.norm(d), "碰撞中心"
    d = np.asarray(joint_b.xyz, dtype=np.float64) - np.asarray(joint_a.xyz, dtype=np.float64)
    if np.linalg.norm(d) > 1e-6:
        return d / np.linalg.norm(d), "关节原点（兜底）"
    return None, "无"


def _pad_face(geo: tuple, inner: np.ndarray, pad_tol_m: float) -> _PadFace:
    """沿 ``inner`` 方向的垫面平面（link 系）。

    ``geo`` 是 ``("box", origin, size)`` 或 ``("mesh", verts)``。网格碰撞下
    垫面层 = 到最内侧表面距离在 ``pad_tol_m`` 以内的顶点；它的切向范围就是
    垫子，而不是整根手指。
    """
    axis = int(np.argmax(np.abs(inner)))
    if abs(inner[axis]) < 0.99:
        raise ValueError(
            f"垫面计算要求内侧方向在 link 系里是轴向的；得到 {inner}。"
            "请吸附到坐标轴或修正 inner_dir_override。"
        )
    tangents = [a for a in range(3) if a != axis]
    if geo[0] == "box":
        origin, size = geo[1], geo[2]
        half = 0.5 * float((size * np.abs(inner)).sum())
        center = origin + inner * half
        face_size = np.asarray(size, dtype=np.float64).copy()
        face_size[axis] = 0.0
        return _PadFace(center=center, size=face_size, axis=axis, d_max=float(center @ inner), slab=None)

    verts = geo[1]
    d = verts @ inner
    d_max = float(d.max())
    slab = verts[d >= d_max - pad_tol_m]
    lo = slab[:, tangents].min(axis=0)
    hi = slab[:, tangents].max(axis=0)
    center = np.zeros(3)  # 用包围盒中心，避免顶点密度把网格拉偏
    center[tangents] = 0.5 * (lo + hi)
    center[axis] = d_max * inner[axis]
    size = np.zeros(3)
    size[tangents] = hi - lo
    return _PadFace(center=center, size=size, axis=axis, d_max=d_max, slab=slab)


def _local_surface_d(face: _PadFace, inner: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """每个探针切向位置下方的表面高度（``v @ inner``）。

    用切平面上最近的垫面顶点；平垫（Panda 就是）处处等于 ``d_max``，曲面
    则跟随表面。层的归属把结果限制在 ``[d_max - pad_tol_m, d_max]``，探针
    永远不会被放到垫面之后。box 面是平的。
    """
    if face.slab is None:
        return np.full(uv.shape[0], face.d_max)
    tangents = [a for a in range(3) if a != face.axis]
    vt = face.slab[:, tangents]
    d = face.slab @ inner
    out = np.empty(uv.shape[0], dtype=np.float64)
    for i, p in enumerate(uv):
        out[i] = d[np.argmin(((vt - p) ** 2).sum(axis=1))]
    return out


def derive_finger_layout(
    urdf_path: str | Path,
    finger_link_names: tuple[str, str],
    *,
    probe_grid: tuple[int, int] = (4, 3),
    probe_offset_m: float = 5e-4,
    margin_frac: float = 0.15,
    pad_tol_m: float = 1.5e-3,
    box_override: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] | None = None,
    inner_dir_override: dict[str, tuple[float, float, float]] | None = None,
    info: dict | None = None,
) -> list[FingerLayout]:
    """在每根手指垫面的内侧面上生成探针网格。

    参数
    ----
    finger_link_names : (左指 link, 右指 link)
        彼此相对的两个 link。闭合轴由共享父节点（手）坐标系里两个碰撞形状
        中心之差得到；没有碰撞几何时回落到指关节原点。
    probe_grid : (沿长度, 沿宽度)
        垫面上的探针数；长度 = 较长的切向尺度。
    probe_offset_m : 探针悬浮在表面之外这么远（接触略早于网格穿透发生）。
    margin_frac : 每个垫面边沿留空的占比。
    pad_tol_m : 网格碰撞的垫面层厚度（见模块 docstring）。
    box_override / inner_dir_override : URDF 几何不可用时的逐 link 校准逃生口。
    info : 可选字典，填入 ``closure_source``、手坐标系下的碰撞中心与垫面
        中心、垫面轴、垫面层厚度（用于自检）。
    """
    names = tuple(finger_link_names)
    meshes_by_link = parse_link_meshes(urdf_path, links=set(names))
    boxes_by_link = parse_link_boxes(urdf_path, links=set(names))
    joints = parse_child_joints(urdf_path)

    ja, jb = (joints.get(name) for name in names)
    if ja is None or jb is None:
        missing = [n for n, j in zip(names, (ja, jb)) if j is None]
        raise ValueError(f"手指 link 不是 URDF 的关节子节点：{missing}")

    geo: dict[str, tuple] = {}
    for name in names:
        if box_override and name in box_override:
            origin, size = box_override[name]
            geo[name] = ("box", np.asarray(origin, dtype=np.float64), np.asarray(size, dtype=np.float64))
        elif name in meshes_by_link:
            geo[name] = ("mesh", meshes_by_link[name])
        elif name in boxes_by_link:
            origin, size = _union_box(boxes_by_link[name])
            geo[name] = ("box", origin, size)
        else:
            raise ValueError(f"{name!r} 没有可用的碰撞几何（box 或 .obj 网格）；请传 box_override")

    def _link_inner(name: str, joint: _Joint, inner_hand: np.ndarray) -> np.ndarray:
        if inner_dir_override and name in inner_dir_override:
            v = np.asarray(inner_dir_override[name], dtype=np.float64)
            return v / np.linalg.norm(v)
        rot = rpy_to_matrix(joint.rpy)  # link 系 -> 父系
        return _snap_to_axis(rot.T @ inner_hand)

    def _hand_center(name: str, joint: _Joint, faces_: dict[str, _PadFace]) -> np.ndarray:
        return joint.xyz + rpy_to_matrix(joint.rpy) @ faces_[name].center

    # 闭合轴取自父（手）坐标系下两个碰撞形状中心之差。这里绝不能用垫面
    # *中心*：夹爪闭合时两个垫面几乎共面（Panda：y = -0.0001 vs +0.0001），
    # 它们之差只有 0.2 mm 的噪声，符号取决于哪个垫面刚好更高——也就是可能
    # 指向远离另一根手指的方向。碰撞中心在 Panda 上相距 26 mm，永远良态。
    centers_hand = {}
    for name, joint in zip(names, (ja, jb)):
        kind, *rest = geo[name]
        c = rest[0] if kind == "box" else 0.5 * (rest[0].min(axis=0) + rest[0].max(axis=0))
        centers_hand[name] = joint.xyz + rpy_to_matrix(joint.rpy) @ c
    axis_ab, closure_source = _closure_direction(centers_hand[names[0]], centers_hand[names[1]], ja, jb)
    if axis_ab is None:
        raise ValueError("无法推导闭合轴：碰撞中心与指关节原点都重合。请传 box_override/inner_dir_override。")
    inner_hand = {names[0]: axis_ab, names[1]: -axis_ab}
    inner_link = {n: _link_inner(n, j, inner_hand[n]) for n, j in zip(names, (ja, jb))}
    faces = {n: _pad_face(geo[n], inner_link[n], pad_tol_m) for n in names}

    if info is not None:
        info["closure_source"] = closure_source
        info["pad_tol_m"] = float(pad_tol_m)
        info["centers_hand"] = {n: v.tolist() for n, v in centers_hand.items()}
        info["pad_face_centers_hand"] = {
            n: _hand_center(n, j, faces).tolist() for n, j in zip(names, (ja, jb))
        }
        info["pad_axes_hand"] = {n: inner_hand[n].tolist() for n in names}

    layouts: list[FingerLayout] = []
    for name in names:
        inner = inner_link[name]
        face = faces[name]

        tangents = [a for a in range(3) if a != face.axis]
        # 长度轴 = 较长的切向尺度。
        tangents.sort(key=lambda a: face.size[a], reverse=True)
        extents = [face.size[a] for a in tangents]
        counts = [probe_grid[0], probe_grid[1]]
        margin = [margin_frac * e for e in extents]

        coord_lists = []
        for extent, margin_a, n in zip(extents, margin, counts):
            usable = max(extent - 2.0 * margin_a, 0.0)
            if n == 1:
                coord_lists.append(np.zeros(1))
            else:
                coord_lists.append(np.linspace(-usable / 2.0, usable / 2.0, n))
        # 网格顺序：第一个切向（长度）为外层循环，第二个（宽度）为内层。
        c_len, c_wid = np.meshgrid(coord_lists[0], coord_lists[1], indexing="ij")
        offsets = np.stack([c_len.ravel(), c_wid.ravel()], axis=1)  # (P, 2)
        uv = offsets + face.center[tangents]  # 垫面上的绝对切向坐标

        d_surface = _local_surface_d(face, inner, uv)
        pos = np.tile(face.center, (offsets.shape[0], 1))
        for k, a in enumerate(tangents):
            pos[:, a] = uv[:, k]
        pos[:, face.axis] = inner[face.axis] * (d_surface + probe_offset_m)

        layouts.append(
            FingerLayout(
                link_name=name,
                probe_local_pos=pos.astype(np.float32),
                probe_local_normal=np.tile(inner.astype(np.float32), (pos.shape[0], 1)),
                pad_box_origin=face.center.astype(np.float32),
                pad_box_size=face.size.astype(np.float32),
                inner_dir=inner.astype(np.float32),
            )
        )
    return layouts


__all__ = [
    "FingerLayout",
    "derive_finger_layout",
    "parse_child_joints",
    "parse_link_boxes",
    "parse_link_meshes",
    "rpy_to_matrix",
]

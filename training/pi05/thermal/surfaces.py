"""候选表面点组装（规格第 6.2 节输入，Task G）。

仿真器/数据集负责给出物体表面点与法向（物体坐标系）。本模块负责：

1. 统一采样到 N 个候选点（不足补零并置 ``mask=False``）；
2. 用点云包围盒中心和尺度把 xyz 归一化（``SurfacePointEncoder`` 的输入约定）；
3. 生成/透传 ``reachable`` 与 ``grasp_score``——仿真器有精确值时优先使用，
   否则用与末端距离、法向-夹爪闭合轴的启发式代替。

这里不依赖 Genesis/trimesh：输入是普通 numpy 点云，便于离线数据与单测。
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class SurfaceCandidates:
    """一批候选表面点（数组均无 batch 维；由数据加载器再堆叠）。"""

    xyz_obj: np.ndarray  # [N, 3] float32，已按点云尺度归一化
    normal_obj: np.ndarray  # [N, 3] float32，单位向量
    mask: np.ndarray  # [N] bool，False 为补齐的无效点
    reachable: np.ndarray  # [N] bool
    grasp_score: np.ndarray  # [N, 1] float32，0..1

    def as_observation_fields(self) -> dict[str, np.ndarray]:
        """与 Observation 字段同名的字典（便于并入数据批次）。"""
        return {
            "surface_xyz_obj": self.xyz_obj,
            "surface_normal_obj": self.normal_obj,
            "surface_mask": self.mask,
            "surface_reachable": self.reachable,
            "surface_grasp_score": self.grasp_score,
        }


def normalize_points(
    points_obj: np.ndarray,
    *,
    center: np.ndarray | None = None,
    scale: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """按包围盒把点归一化到约 [-1, 1]；返回 (归一化点, 中心, 尺度)。"""
    points = np.asarray(points_obj, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError(f"points_obj 必须为 [P, 3] 且 P>0，得到 {points.shape}")
    if center is None:
        center = 0.5 * (points.min(axis=0) + points.max(axis=0))
    center = np.asarray(center, dtype=np.float32).reshape(3)
    if scale is None:
        extent = points.max(axis=0) - points.min(axis=0)
        scale = float(np.max(extent)) * 0.5
    scale = max(float(scale), 1e-6)
    return (points - center) / scale, center, scale


def _unit_normals(normals: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    # 零法向在无效点上允许存在；有效点由调用方保证非零。
    safe = np.where(norm < 1e-8, 1.0, norm)
    return (normals / safe).astype(np.float32)


def _as_column(values: np.ndarray, length: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.shape[0] != length:
        raise ValueError(f"{name} 长度必须为 {length}，得到 {array.shape[0]}")
    return array


def build_surface_candidates(
    points_obj: np.ndarray,
    normals_obj: np.ndarray,
    ee_xyz_obj: np.ndarray,
    *,
    num_points: int,
    rng: np.random.Generator | None = None,
    reach_radius_m: float = 0.25,
    grasp_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    reachable: np.ndarray | None = None,
    grasp_score: np.ndarray | None = None,
    center: np.ndarray | None = None,
    scale: float | None = None,
) -> SurfaceCandidates:
    """从稠密表面点云构造 N 个候选点。

    Args:
        points_obj / normals_obj: [P, 3] 物体坐标系表面点与法向。
        ee_xyz_obj: [3] 末端位置（物体坐标系），用于 reachable 启发式与距离特征。
        num_points: 候选点数量 N。
        rng: 采样随机源；None 时用等距索引（确定性、无随机流副作用）。
        reach_radius_m: 启发式可达半径。
        grasp_axis: 夹爪闭合轴（物体坐标系），用于 grasp_score 启发式。
        reachable / grasp_score: 仿真器的精确标签（长度 P），提供时优先。
    """
    points = np.asarray(points_obj, dtype=np.float32)
    normals = np.asarray(normals_obj, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_obj 必须为 [P, 3]，得到 {points.shape}")
    if normals.shape != points.shape:
        raise ValueError(f"normals_obj 形状必须与 points_obj 相同，得到 {normals.shape} 与 {points.shape}")
    if num_points < 1:
        raise ValueError("num_points 必须 >= 1")
    point_count = points.shape[0]
    ee = np.asarray(ee_xyz_obj, dtype=np.float32).reshape(3)

    # 1) 采样索引；不足 N 时用重复索引补齐，随后由 mask 标为无效。
    if point_count >= num_points:
        if rng is None:
            indices = np.linspace(0, point_count - 1, num_points).round().astype(int)
        else:
            indices = rng.choice(point_count, num_points, replace=False)
    else:
        indices = np.concatenate([np.arange(point_count), np.zeros(num_points - point_count, dtype=int)])
    valid = np.zeros((num_points,), dtype=bool)
    valid[: min(point_count, num_points)] = True

    normalized, _, _ = normalize_points(points, center=center, scale=scale)
    xyz = normalized[indices].astype(np.float32)
    normal = _unit_normals(normals[indices])
    # 补齐位（mask=False）的几何量置零，避免编码器读到重复点的值。
    xyz[~valid] = 0.0
    normal[~valid] = 0.0

    # 2) reachable / grasp_score：精确标签优先，否则用启发式。
    if reachable is None:
        distance = np.linalg.norm(points - ee[None, :], axis=-1)
        reachable_values = (distance <= float(reach_radius_m)).astype(bool)
    else:
        reachable_values = _as_column(reachable, point_count, "reachable") > 0.5
    reachable_values = reachable_values[indices] & valid

    if grasp_score is None:
        axis = np.asarray(grasp_axis, dtype=np.float32).reshape(3)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-8:
            raise ValueError("grasp_axis 不能为零向量")
        axis = axis / axis_norm
        # 接触法向与夹爪闭合轴越对齐，越接近对跖抓取；仅作占位启发式。
        scores = np.clip(np.abs(normals @ axis), 0.0, 1.0).astype(np.float32)
    else:
        scores = _as_column(grasp_score, point_count, "grasp_score")
    scores = np.where(valid, scores[indices], 0.0).astype(np.float32)

    return SurfaceCandidates(
        xyz_obj=xyz,
        normal_obj=normal,
        mask=valid,
        reachable=reachable_values.astype(bool),
        grasp_score=scores.reshape(-1, 1),
    )


__all__ = ["SurfaceCandidates", "build_surface_candidates", "normalize_points"]

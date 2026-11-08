"""热区与 link 的映射。

资产清单（``assets/cup_4link/manifest.yaml``）是*唯一*真源：区域顺序、
区域对应的 link 名、区域 z 边界都来自清单，代码里不写死任何 link 名。

``ThermalRegionRuntime`` 把清单解析成运行时可用的表：

* ``region_id`` —— 采样、记录、标签所用的整数编号（等于 ``region_order`` 下标）；
* ``link_name`` —— 资产里的 link 名（``entity.links`` 的键）；
* ``entity_link_idx`` —— 该 link 在 ``entity.links`` 里的局部下标；
* ``global_link_idx`` —— 场景级 link 下标，``link_temps`` 表的列号；
* ``z_span_m`` —— 该区域在杯子作者坐标系里的 z 范围（用于校验探测点）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class ThermalRegion:
    """单个热区。不可变：区域表在场景构建时一次成型。"""

    region_id: int
    name: str
    link_name: str
    entity_link_idx: int
    global_link_idx: int
    z_span_m: tuple[float, float]


@dataclass(frozen=True)
class ThermalRegionRuntime:
    """四个热区的有序表（顺序 = 清单的 ``region_order``）。"""

    regions: tuple[ThermalRegion, ...]
    order: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.regions)

    def __iter__(self):
        return iter(self.regions)

    # ------------------------------------------------------------ 按名字/编号
    def by_id(self, region_id: int) -> ThermalRegion:
        if not 0 <= region_id < len(self.regions):
            raise IndexError(f"region_id {region_id} 超出 [0, {len(self.regions)})")
        return self.regions[region_id]

    def by_name(self, name: str) -> ThermalRegion:
        for region in self.regions:
            if region.name == name:
                return region
        raise KeyError(f"未知热区 {name!r}；可用：{list(self.order)}")

    # ---------------------------------------------------------------- 批量视图
    @property
    def link_names(self) -> tuple[str, ...]:
        return tuple(r.link_name for r in self.regions)

    @property
    def entity_link_indices(self) -> np.ndarray:
        return np.asarray([r.entity_link_idx for r in self.regions], dtype=np.int64)

    @property
    def global_link_indices(self) -> np.ndarray:
        """``link_temps`` 的列号向量，形状 (4,)。"""
        return np.asarray([r.global_link_idx for r in self.regions], dtype=np.int64)

    @property
    def z_spans(self) -> np.ndarray:
        return np.asarray([r.z_span_m for r in self.regions], dtype=np.float64)

    # ------------------------------------------------------------ link -> 区域
    def region_of_link(self, global_link_idx: int) -> int:
        """场景级 link 下标 -> region_id；不属于杯子的 link 返回 -1。"""
        for region in self.regions:
            if region.global_link_idx == int(global_link_idx):
                return region.region_id
        return -1

    def region_of_contact(self, link_a: Iterable[int], link_b: Iterable[int]) -> np.ndarray:
        """把接触对映射成 region_id 数组（任一侧属于杯子即可，否则 -1）。"""
        a = np.asarray(list(link_a), dtype=np.int64)
        b = np.asarray(list(link_b), dtype=np.int64)
        if a.shape != b.shape:
            raise ValueError(f"link_a / link_b 形状不一致：{a.shape} vs {b.shape}")
        out = np.full(a.shape, -1, dtype=np.int64)
        for region in self.regions:
            hit = (a == region.global_link_idx) | (b == region.global_link_idx)
            out[hit] = region.region_id
        return out


def resolve_thermal_regions(entity: Any, object_cfg: dict[str, Any]) -> ThermalRegionRuntime:
    """按清单把实体 link 解析成热区表。

    要求实体恰好包含清单列出的四个 link（``merge_fixed_links=False`` 下
    四段杯子就是四个 link），否则立刻报错——静默错配会让温度读数和接触
    归属整体错位。
    """
    order: Sequence[str] = object_cfg["region_order"]
    links = object_cfg["region_links"]
    spans = object_cfg["region_z_spans"]

    by_name: dict[str, Any] = {}
    for link in entity.links:
        by_name[str(link.name)] = link
    missing = [links[name] for name in order if links[name] not in by_name]
    if missing:
        raise ValueError(f"杯子实体缺少 link：{missing}（实际 link：{sorted(by_name)}）")
    if len(by_name) != len(order):
        raise ValueError(
            f"杯子实体应有 {len(order)} 个 link，实际 {len(by_name)} 个：{sorted(by_name)}"
        )

    regions = []
    for i, name in enumerate(order):
        link = by_name[links[name]]
        idx = link.idx
        idx_np = np.asarray(idx.detach().cpu() if hasattr(idx, "detach") else idx).reshape(-1)
        if idx_np.size != 1:
            raise ValueError(f"link {links[name]} 的 idx 不是单值：{idx_np}")
        lo, hi = (float(v) for v in spans[name])
        regions.append(
            ThermalRegion(
                region_id=i,
                name=str(name),
                link_name=str(links[name]),
                entity_link_idx=_link_position(entity, links[name]),
                global_link_idx=int(idx_np[0]),
                z_span_m=(lo, hi),
            )
        )
    # 各区域 z 范围只要求落在资产 AABB 内：前三个区域沿 z 堆叠，手柄是与
    # 它们并行的独立分支（z 范围故意重叠），所以不能做单调性检查。
    z_lo = float(object_cfg["aabb_min_m"][2]) - 1e-6
    z_hi = float(object_cfg["aabb_max_m"][2]) + 1e-6
    for region in regions:
        lo, hi = region.z_span_m
        if not z_lo <= lo < hi <= z_hi:
            raise ValueError(
                f"区域 {region.name} 的 z 范围 [{lo}, {hi}] 超出资产 z 范围 [{z_lo}, {z_hi}]"
            )
    return ThermalRegionRuntime(regions=tuple(regions), order=tuple(str(n) for n in order))


def _link_position(entity: Any, link_name: str) -> int:
    """link 在 ``entity.links`` 里的局部下标。"""
    for i, link in enumerate(entity.links):
        if str(link.name) == link_name:
            return i
    raise ValueError(f"link {link_name} 不在 entity.links 中")


__all__ = ["ThermalRegion", "ThermalRegionRuntime", "resolve_thermal_regions"]

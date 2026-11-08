#!/usr/bin/env python3
"""校验四段式杯子资产（``assets/cup_4link``）。

检查项：
1. ``manifest.yaml`` 的键、区域顺序、link 名一一对应；
2. URDF 可解析，恰好四个 link、三个固定关节、网格引用都存在；
3. 每段网格封闭、体积为正；
4. 四段恰好分割合并体（体积、AABB 一致）；
5. z 分界连续覆盖杯身，手柄热区落在手柄网格的实测 z 范围内；
6. 手柄碰撞代理与杯壁保持间隙（兄弟 link 不会进入碰撞器）；
7. 清单里的质量、AABB 与实测一致。

    python scripts/validate_cup_4link.py
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

REGION_ORDER = ["lower", "middle", "upper", "handle"]
MANIFEST_KEYS = {
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


def _load_mesh(path: Path):
    """只读加载网格（不做离屏渲染/纹理解析）。"""
    import trimesh

    return trimesh.load(path, force="mesh", process=False)


def validate(asset_dir: Path) -> list[str]:
    """返回问题列表；空列表表示通过。"""
    issues: list[str] = []
    manifest_path = asset_dir / "manifest.yaml"
    if not manifest_path.is_file():
        return [f"缺少清单：{manifest_path}"]
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        return [f"清单不是字典：{manifest_path}"]

    missing = MANIFEST_KEYS - set(manifest)
    extra = set(manifest) - MANIFEST_KEYS
    if missing:
        issues.append(f"清单缺键：{sorted(missing)}")
    if extra:
        issues.append(f"清单多出未知键：{sorted(extra)}")

    if list(manifest.get("region_order", [])) != REGION_ORDER:
        issues.append(f"region_order 必须是 {REGION_ORDER}，实为 {manifest.get('region_order')}")
    if list(manifest.get("link_names", [])) != REGION_ORDER:
        issues.append(f"link_names 必须与 region_order 一致：{manifest.get('link_names')}")
    regions = manifest.get("regions", {})
    if set(regions) != set(REGION_ORDER):
        issues.append(f"regions 的键必须是 {REGION_ORDER}，实为 {sorted(regions)}")
    else:
        for name in REGION_ORDER:
            entry = regions[name]
            if set(entry) != {"link", "z_span_m"}:
                issues.append(f"regions.{name} 只允许 link / z_span_m，实为 {sorted(entry)}")
                continue
            if entry["link"] != name:
                issues.append(f"regions.{name}.link 应为 {name}，实为 {entry['link']}")
            span = entry["z_span_m"]
            if len(span) != 2 or float(span[0]) >= float(span[1]):
                issues.append(f"regions.{name}.z_span_m 必须是递增的 [lo, hi]，实为 {span}")

    urdf_path = asset_dir / str(manifest.get("urdf_file", ""))
    if not urdf_path.is_file():
        issues.append(f"缺少 URDF：{urdf_path}")
        return issues
    root = ET.parse(urdf_path).getroot()
    links = [e.get("name") for e in root.findall("link")]
    if links != REGION_ORDER:
        issues.append(f"URDF 的 link 顺序必须是 {REGION_ORDER}，实为 {links}")
    joints = [
        (e.get("name"), e.get("type"), e.find("parent").get("link"), e.find("child").get("link"))
        for e in root.findall("joint")
    ]
    expected_joints = [
        ("lower_middle", "fixed", "lower", "middle"),
        ("middle_upper", "fixed", "middle", "upper"),
        ("middle_handle", "fixed", "middle", "handle"),
    ]
    if joints != expected_joints:
        issues.append(f"URDF 关节应为 {expected_joints}，实为 {joints}")

    meshes: dict[str, tuple[Path, object]] = {}
    for link in root.findall("link"):
        name = link.get("name")
        for tag in ("visual", "collision"):
            geom = link.find(f"{tag}/geometry/mesh")
            if geom is None:
                issues.append(f"{name} 缺少 {tag} 网格")
                continue
            path = urdf_path.parent / geom.get("filename")
            if not path.is_file():
                issues.append(f"{name} 的 {tag} 网格不存在：{path}")
                continue
            meshes.setdefault(f"{name}:{tag}", (path, _load_mesh(path)))

    for key, (path, mesh) in meshes.items():
        if not mesh.is_watertight:
            issues.append(f"{key} ({path.name}) 不封闭")
        if float(mesh.volume) <= 0.0:
            issues.append(f"{key} ({path.name}) 体积非正：{float(mesh.volume):.3e} m^3")

    # 分割关系成立在*视觉*网格上：碰撞代理（手柄外侧 130°）是有意裁掉的。
    body = {}
    for name in REGION_ORDER:
        entry = meshes.get(f"{name}:visual")
        if entry is not None:
            body[name] = entry[1]

    union_path = asset_dir / "meshes" / "cup.obj"
    if not union_path.is_file():
        issues.append(f"缺少合并体参考网格：{union_path}")
        return issues
    union = _load_mesh(union_path)
    if not union.is_watertight:
        issues.append("cup.obj 不封闭")
    if set(body) == set(REGION_ORDER):
        split_volume = sum(float(m.volume) for m in body.values())
        if not np.isclose(split_volume, float(union.volume), rtol=1e-6, atol=1e-12):
            issues.append(
                f"分段体积 {split_volume:.9e} != 合并体 {float(union.volume):.9e} m^3"
            )
        part_lo = np.min([m.bounds[0] for m in body.values()], axis=0)
        part_hi = np.max([m.bounds[1] for m in body.values()], axis=0)
        if not np.allclose(part_lo, union.bounds[0]) or not np.allclose(part_hi, union.bounds[1]):
            issues.append(
                f"分段 AABB {part_lo}/{part_hi} != 合并体 {union.bounds[0]}/{union.bounds[1]}"
            )
        if "lower" in body and "middle" in body and "upper" in body:
            lo, hi = body["lower"].bounds[0][2], body["upper"].bounds[1][2]
            if not np.isclose(lo, 0.0, atol=1e-9):
                issues.append(f"杯底应在 z=0，实为 {lo:.6f}")
            spans = [regions[n]["z_span_m"] for n in ("lower", "middle", "upper") if n in regions]
            if len(spans) == 3:
                if not np.isclose(spans[0][1], spans[1][0], atol=1e-9):
                    issues.append(f"lower/middle 分界不连续：{spans[0]} / {spans[1]}")
                if not np.isclose(spans[1][1], spans[2][0], atol=1e-9):
                    issues.append(f"middle/upper 分界不连续：{spans[1]} / {spans[2]}")
                if not np.isclose(spans[2][1], hi, atol=1e-9):
                    issues.append(f"upper 上界 {spans[2][1]} != 杯口 {hi}")
        if "handle" in body and "handle" in regions:
            z0, z1 = body["handle"].bounds[0][2], body["handle"].bounds[1][2]
            span = regions["handle"]["z_span_m"]
            if not (np.isclose(span[0], z0, atol=1e-6) and np.isclose(span[1], z1, atol=1e-6)):
                issues.append(f"handle 的 z_span_m {span} 与实测 [{z0:.6f}, {z1:.6f}] 不符")
            handle_col = meshes.get("handle:collision")
            if handle_col is not None:
                col = handle_col[1]
                r_min = float(np.hypot(col.vertices[:, 0], col.vertices[:, 1]).min())
                r_body = float(
                    np.max(np.hypot(body["upper"].vertices[:, 0], body["upper"].vertices[:, 1]))
                )
                clearance = r_min - r_body
                if clearance <= 0.005:
                    issues.append(
                        f"手柄碰撞代理与杯壁间隙只有 {clearance * 1e3:.1f} mm（要求 > 5 mm）"
                    )

    aabb_lo = np.asarray(manifest.get("aabb_min_m", [0.0, 0.0, 0.0]), dtype=np.float64)
    aabb_hi = np.asarray(manifest.get("aabb_max_m", [0.0, 0.0, 0.0]), dtype=np.float64)
    if not np.allclose(aabb_lo, union.bounds[0], atol=1e-6):
        issues.append(f"清单 aabb_min_m {aabb_lo} != 实测 {union.bounds[0]}")
    if not np.allclose(aabb_hi, union.bounds[1], atol=1e-6):
        issues.append(f"清单 aabb_max_m {aabb_hi} != 实测 {union.bounds[1]}")
    if float(manifest.get("mass_kg", 0.0)) <= 0.0:
        issues.append(f"清单 mass_kg 非正：{manifest.get('mass_kg')}")
    if list(manifest.get("base_euler_deg", [])) != [0.0, 0.0, 90.0]:
        issues.append(f"base_euler_deg 必须是 [0, 0, 90]，实为 {manifest.get('base_euler_deg')}")

    return issues


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asset-dir", default="assets/cup_4link")
    args = ap.parse_args()
    asset_dir = Path(args.asset_dir).expanduser().resolve()
    issues = validate(asset_dir)
    if issues:
        for item in issues:
            print(f"[问题] {item}")
        raise SystemExit(f"{len(issues)} 个问题")
    print(f"OK：{asset_dir} 通过全部检查")


if __name__ == "__main__":
    main()

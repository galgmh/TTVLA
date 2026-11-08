"""区域映射测试：清单 -> 运行时区域表 -> 接触归属。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests._scene import get_bundle, task_config  # noqa: E402


class RegionMappingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = get_bundle()
        cls.regions = cls.bundle.regions
        cls.task = task_config()

    def test_order_matches_manifest(self) -> None:
        manifest_order = tuple(self.task["object"]["region_order"])
        self.assertEqual(self.regions.order, manifest_order)
        self.assertEqual(len(self.regions), 4)
        self.assertEqual([r.name for r in self.regions], list(manifest_order))

    def test_global_indices_are_cup_links(self) -> None:
        cup_indices = {int(link.idx) for link in self.bundle.cup.links}
        self.assertEqual(set(int(i) for i in self.regions.global_link_indices), cup_indices)
        # link_temp 的列顺序 = region_order
        self.assertEqual(self.regions.global_link_indices.shape, (4,))

    def test_region_of_link(self) -> None:
        for region in self.regions:
            self.assertEqual(self.regions.region_of_link(region.global_link_idx), region.region_id)
        robot_link = int(self.bundle.robot.links[0].idx)
        self.assertEqual(self.regions.region_of_link(robot_link), -1)

    def test_contact_pairs_map_both_sides(self) -> None:
        lower = int(self.regions.by_name("lower").global_link_idx)
        robot_link = int(self.bundle.robot.links[0].idx)
        mapped = self.regions.region_of_contact([robot_link], [lower])
        self.assertEqual(mapped.tolist(), [0])
        # 反向（杯子在前）也要识别。
        mapped = self.regions.region_of_contact([lower], [robot_link])
        self.assertEqual(mapped.tolist(), [0])
        mapped = self.regions.region_of_contact([robot_link], [robot_link])
        self.assertEqual(mapped.tolist(), [-1])

    def test_z_spans_within_asset(self) -> None:
        aabb_lo = float(np.min([r.z_span_m[0] for r in self.regions]))
        aabb_hi = float(np.max([r.z_span_m[1] for r in self.regions]))
        self.assertGreaterEqual(aabb_lo, -1e-9)
        self.assertLess(aabb_hi, 0.2)


if __name__ == "__main__":
    unittest.main()

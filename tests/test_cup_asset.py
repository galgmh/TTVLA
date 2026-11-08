"""资产测试：清单与 URDF 的一致性、区域 z 分界。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.validate_cup_4link import validate  # noqa: E402
from tests._scene import REPO_ROOT  # noqa: E402

ASSET_DIR = REPO_ROOT / "assets" / "cup_4link"


class CupAssetTest(unittest.TestCase):
    def test_asset_passes_validator(self) -> None:
        issues = validate(ASSET_DIR)
        self.assertEqual(issues, [], "\n".join(issues))

    def test_region_z_spans_cover_body(self) -> None:
        import yaml

        manifest = yaml.safe_load((ASSET_DIR / "manifest.yaml").read_text(encoding="utf-8"))
        spans = manifest["regions"]
        # 前三个区域沿 z 首尾相接，覆盖杯身全高。
        body = ["lower", "middle", "upper"]
        for prev, nxt in zip(body, body[1:]):
            self.assertAlmostEqual(spans[prev]["z_span_m"][1], spans[nxt]["z_span_m"][0], places=9)
        lo = manifest["aabb_min_m"][2]
        hi = manifest["aabb_max_m"][2]
        self.assertAlmostEqual(spans["lower"]["z_span_m"][0], lo, places=9)
        self.assertAlmostEqual(spans["upper"]["z_span_m"][1], hi, places=9)


if __name__ == "__main__":
    unittest.main()

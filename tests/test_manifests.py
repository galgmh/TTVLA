"""清单一致性测试：冻结的 manifest 必须与采样器逐字段一致。

manifest 是“每集完整参数组合”的唯一真源；生成脚本用同一采样器复现。
这里抽查若干集，确保 ``data/manifests`` 里的规格没有被代码或种子漂移改掉。
清单不存在时跳过（清单不入库）。
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.config import load_dataset_config, load_task_config  # noqa: E402
from sim.sampling.thermal_contact_sampler import ThermalContactSampler  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = REPO_ROOT / "data" / "manifests"
SPLITS = ("pilot", "validation", "test_mixed_contact")


def _load_manifest(split: str) -> list[dict]:
    path = MANIFEST_DIR / f"thermalcontact-v1_{split}.jsonl"
    if not path.exists():
        raise unittest.SkipTest(f"清单不存在：{path}")
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise unittest.SkipTest(f"清单为空：{path}")
    return lines


class ManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.task = load_task_config(str(REPO_ROOT / "configs/tasks/thermal_contact_probe.yaml"))
        cls.dataset = load_dataset_config(str(REPO_ROOT / "configs/datasets/thermal_contact_v1.yaml"))

    def _check(self, split: str) -> None:
        rows = _load_manifest(split)
        sampler = ThermalContactSampler(self.task, self.dataset, split)
        picks = sorted({0, len(rows) // 2, len(rows) - 1})
        for index in picks:
            spec = sampler.sample(index)
            self.assertEqual(rows[index]["index"], index)
            self.assertEqual(rows[index]["spec"], spec.to_dict(), f"{split}#{index} 与清单不一致")

    def test_pilot_matches(self) -> None:
        self._check("pilot")

    def test_validation_matches(self) -> None:
        self._check("validation")

    def test_mixed_contact_matches(self) -> None:
        self._check("test_mixed_contact")


if __name__ == "__main__":
    unittest.main()

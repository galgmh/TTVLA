"""纯接触测试：跑一整集受控触碰，检查接触归属、力与外来接触。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.control.probe_controller import ProbeController  # noqa: E402
from sim.obs import contact_link_mask, get_privileged_observation  # noqa: E402
from sim.sampling.thermal_contact_sampler import ThermalContactSampler  # noqa: E402
from tests._scene import dataset_config, get_bundle, task_config  # noqa: E402


class PureRegionContactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = get_bundle()
        cls.task = task_config()
        cls.sampler = ThermalContactSampler(cls.task, dataset_config(), "pilot")
        # 找一个 lower 稳态集：pilot 按 4 区域轮转，质量随机，取前 40 集即可。
        cls.spec = None
        for index in range(40):
            spec = cls.sampler.sample(index)
            if spec.target_region == 0 and spec.contact_quality == "stable":
                cls.spec = spec
                break
        if cls.spec is None:
            raise unittest.SkipTest("前 40 集里没有 lower 稳定接触样本")

    def test_lower_probe_is_pure(self) -> None:
        bundle = self.bundle
        spec = self.spec
        bundle.reset_episode(spec)
        probe = ProbeController(bundle, spec)
        mask, foreign = contact_link_mask(bundle)
        self.assertFalse(mask.any(), "复位后不应有接触")
        self.assertFalse(foreign)

        seen_masks: list[np.ndarray] = []
        seen_foreign = [False]
        seen_peak = [0.0]

        def on_sample(sim_step: int) -> None:
            if probe.phase != "HOLD":
                return
            m, f = contact_link_mask(bundle)
            if m.any():
                seen_masks.append(m.copy())
                seen_foreign[0] = seen_foreign[0] or f
            seen_peak[0] = max(seen_peak[0], float(get_privileged_observation(bundle)["link_temp"][0]))

        outcome = probe.run(on_sample=on_sample)
        self.assertTrue(outcome.contact_detected, f"未建立接触：{outcome.termination_reason}")
        self.assertGreater(outcome.contact_end_s or 0.0, outcome.contact_start_s or 0.0)
        self.assertGreater(outcome.peak_normal_force_n, 0.5)
        self.assertLess(outcome.peak_normal_force_n, 8.0)
        self.assertTrue(seen_masks, "HOLD 期间没有记录到接触")
        self.assertFalse(seen_foreign[0], "出现手指以外的机器人 link 接触杯子")

        # 每帧只允许目标区域（lower）被碰到。
        for m in seen_masks:
            regions = np.where(m.any(axis=0))[0]
            self.assertEqual(regions.tolist(), [0], f"接触区域应为 [0]，实为 {regions.tolist()}")

        # 杯子几乎没有位移：单区域触碰不是抓取。
        bundle.reset_episode(spec)
        before = np.asarray(bundle.cup.get_pos().detach().cpu()).reshape(-1)[:3].copy()
        probe = ProbeController(bundle, spec)
        probe.run()
        after = np.asarray(bundle.cup.get_pos().detach().cpu()).reshape(-1)[:3].copy()
        self.assertLess(float(np.linalg.norm(after - before)), 0.005)


if __name__ == "__main__":
    unittest.main()

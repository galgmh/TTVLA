"""温度播种测试：复位后四段 link 温度与指尖初始温度必须等于本集规格。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.sampling.thermal_contact_sampler import ThermalContactSampler  # noqa: E402
from sim.thermal.link_temperature import read_region_temperatures  # noqa: E402
from sim.sensors.temperature import read_temperature  # noqa: E402
from tests._scene import dataset_config, get_bundle, task_config  # noqa: E402


class LinkTemperatureSeedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = get_bundle()
        cls.sampler = ThermalContactSampler(task_config(), dataset_config(), "pilot")

    def test_seed_matches_spec(self) -> None:
        bundle = self.bundle
        for index in (0, 5, 10):
            spec = self.sampler.sample(index)
            bundle.reset_episode(spec)
            table = read_region_temperatures(bundle.temperatures[0], bundle.regions)
            values = np.asarray(table.detach().cpu()).reshape(-1)
            np.testing.assert_allclose(values, np.asarray(spec.link_temps_c), atol=1e-4)

            # 真值缓冲：直接等于本集规格。
            truth = read_temperature(bundle.temperatures, ground_truth=True)
            finger = np.asarray(
                [
                    np.asarray(truth[h.name].detach().cpu()).reshape(-1)[0]
                    for h in bundle.temperatures
                ],
                dtype=np.float64,
            )
            np.testing.assert_allclose(finger, np.asarray(spec.fingertip_temps_c), atol=1e-3)

            # 测量通道也初始化在播种值：复位后第一帧读数不需要收敛过渡，
            # 只差噪声/偏置/量化（偏置 ≤0.01 °C、噪声 σ≈0.05 °C、量化 0.1 °C）。
            measured = read_temperature(bundle.temperatures)
            finger_measured = np.asarray(
                [
                    np.asarray(measured[h.name].detach().cpu()).reshape(-1)[0]
                    for h in bundle.temperatures
                ],
                dtype=np.float64,
            )
            np.testing.assert_allclose(
                finger_measured, np.asarray(spec.fingertip_temps_c), atol=0.3
            )

    def test_seed_is_not_global(self) -> None:
        """换一集必须换温度：复位会先写回基准温度再播种。"""
        bundle = self.bundle
        first = self.sampler.sample(0)
        second = self.sampler.sample(1)
        bundle.reset_episode(first)
        a = np.asarray(read_region_temperatures(bundle.temperatures[0], bundle.regions).detach().cpu()).reshape(-1)
        bundle.reset_episode(second)
        b = np.asarray(read_region_temperatures(bundle.temperatures[0], bundle.regions).detach().cpu()).reshape(-1)
        self.assertFalse(np.allclose(a, b))
        np.testing.assert_allclose(b, np.asarray(second.link_temps_c), atol=1e-4)


if __name__ == "__main__":
    unittest.main()

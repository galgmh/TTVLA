"""复位测试：每集复位必须回到初始位姿、按规格摆放杯子并重播温度。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.sampling.thermal_contact_sampler import ThermalContactSampler  # noqa: E402
from tests._scene import dataset_config, get_bundle, task_config  # noqa: E402


class EpisodeResetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = get_bundle()
        cls.task = task_config()
        cls.sampler = ThermalContactSampler(cls.task, dataset_config(), "pilot")

    def test_reset_restores_initial_state(self) -> None:
        bundle = self.bundle
        spec = self.sampler.sample(2)
        bundle.reset_episode(spec)
        q_init = np.asarray(self.task["robot"]["init_qpos"], dtype=np.float64)
        q0 = bundle.joint_state()
        np.testing.assert_allclose(q0, q_init, atol=1e-6)

        pos_a, quat_a = bundle.cup_pose()
        # 推进若干仿真步后重新复位：位姿与关节都必须回到本集初始状态。
        for _ in range(5):
            bundle.step()
        bundle.reset_episode(spec)
        np.testing.assert_allclose(bundle.joint_state(), q0, atol=1e-6)
        pos_b, quat_b = bundle.cup_pose()
        np.testing.assert_allclose(pos_b, pos_a, atol=1e-9)
        np.testing.assert_allclose(quat_b, quat_a, atol=1e-9)

    def test_cup_pose_follows_spec_jitter(self) -> None:
        bundle = self.bundle
        poses = []
        for index in range(4):
            spec = self.sampler.sample(index)
            bundle.reset_episode(spec)
            poses.append(bundle.cup_pose()[0])
        stacked = np.stack(poses)
        # 不同集的抖动不同，位置必须真的变化（但幅度受配置约束）。
        self.assertGreater(float(np.ptp(stacked[:, 0])), 0.0)
        self.assertLess(float(np.max(np.abs(stacked[:, 0] - stacked[:, 0].mean()))), 0.02)

    def test_reset_does_not_move_robot_into_cup(self) -> None:
        """复位后 TCP 必须离杯口足够远：复位本身不能产生接触。"""
        bundle = self.bundle
        spec = self.sampler.sample(1)
        bundle.reset_episode(spec)
        contacts = bundle.cup.get_contacts()
        link_a = np.asarray(contacts["link_a"].detach().cpu()).reshape(-1).astype(int) if contacts else np.zeros(0, int)
        link_b = np.asarray(contacts["link_b"].detach().cpu()).reshape(-1).astype(int) if contacts else np.zeros(0, int)
        robot_links = {int(link.idx) for link in bundle.robot.links}
        touching = [i for i in range(len(link_a)) if int(link_a[i]) in robot_links or int(link_b[i]) in robot_links]
        self.assertEqual(touching, [], "复位位姿下机器人不应接触杯子")


if __name__ == "__main__":
    unittest.main()

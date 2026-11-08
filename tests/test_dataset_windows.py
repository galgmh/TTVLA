"""窗口加载测试：用合成 episode 检查切窗、标签与归一化。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.datasets.thermal_contact_dataset import (  # noqa: E402
    build_windows,
    compute_stats,
    load_episode,
    window_end_times,
)
from data.datasets.window_dataset import ThermalWindowDataset, compute_stats as stats_stream  # noqa: E402
from data.schema.thermal_contact_v1 import ENCODER_INPUT_CHANNELS, empty_episode  # noqa: E402
from data.writer.hdf5_writer import write_episode  # noqa: E402

PREFIXES = [0.10, 0.20, 0.30, 0.50]


def _write_episode(path: Path, *, target_region: int, contact_start: float, contact_end: float, link_temps):
    """写一集 2 s / 100 Hz 的合成数据：接触段由时间决定。"""
    buffers = empty_episode()
    dt = 0.01
    frames = 200
    for i in range(frames):
        t = i * dt
        inside = contact_start <= t <= contact_end
        temp = 25.0 + 0.5 * np.tanh(3.0 * (t - contact_start)) * (link_temps[target_region] - 25.0) / 30.0
        mask = np.zeros((2, 4), dtype=bool)
        if inside and target_region >= 0:
            mask[0, target_region] = True
        buffers["time/sim_time"].append(t)
        buffers["obs/fingertip_temp"].append(np.array([temp, temp + 0.4]))
        buffers["obs/contact_force_world"].append(np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]) * inside)
        buffers["obs/contact_normal_force"].append(np.array([1.5, 1.2]) * inside)
        buffers["obs/contact_mask"].append(np.array([inside, inside]))
        buffers["obs/gripper_width"].append(0.06)
        buffers["obs/finger_joint_position"].append(np.array([0.03, 0.03]))
        buffers["obs/eef_pose"].append(np.array([0.3, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]))
        buffers["obs/contact_area"].append(np.array([1e-4, 1e-4]) * inside)
        buffers["action/eef_target"].append(np.array([0.3, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]))
        buffers["action/gripper_target"].append(0.06)
        buffers["action/target_force"].append(2.0)
        buffers["action/probe_region"].append(target_region)
        buffers["action/phase"].append(4 if inside else 2)
        buffers["privileged/link_temp"].append(np.asarray(link_temps, dtype=np.float64))
        buffers["privileged/contact_link_mask"].append(mask)
        buffers["privileged/pure_contact"].append(bool(inside) and target_region >= 0)
        buffers["privileged/contact_valid"].append(bool(inside))
    metadata = {
        "episode_id": path.stem,
        "dataset_version": "thermalcontact-v1",
        "random_seed": 0,
        "split": "pilot",
        "region_order": ["lower", "middle", "upper", "handle"],
        "target_region": target_region,
        "regime": "case_1",
        "finger_mode": "ambient",
        "cup_pose": [0.0] * 7,
        "cup_asset_version": "cup4link-v1",
        "material_parameters": {"conductivity": 1.5},
        "sensor_RC": 0.05,
        "sensor_noise": 0.05,
        "sensor_bias": 0.0,
        "quantization": 0.1,
        "target_force_level": 1.0,
        "target_force_n": 2.0,
        "initial_finger_temperature": [25.0, 25.4],
        "link_temperature": [float(v) for v in link_temps],
        "contact_quality": "stable",
        "contact_hold_s": 0.5,
        "contact_start_time": contact_start,
        "contact_end_time": contact_end,
        "termination_reason": "contact_stable",
        "policy_steps": 100,
        "peak_normal_force_n": 1.5,
    }
    scalars = {"privileged/target_region": target_region}
    write_episode(path, buffers, metadata, scalars)


class WindowLoaderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.files = []
        cls.files.append(root / "a.hdf5")
        _write_episode(
            cls.files[0],
            target_region=1,
            contact_start=1.0,
            contact_end=1.5,
            link_temps=[20.0, 35.0, 50.0, 25.0],
        )
        cls.failed = root / "b.hdf5"
        _write_episode(
            cls.failed,
            target_region=2,
            contact_start=99.0,
            contact_end=99.0,
            link_temps=[20.0, 35.0, 50.0, 25.0],
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_frame_features_channels(self) -> None:
        episode = load_episode(self.files[0])
        windows = build_windows(episode, prefixes_s=PREFIXES, training_hz=50.0, stride=2)
        self.assertEqual(windows.features.shape[-1], len(ENCODER_INPUT_CHANNELS))
        self.assertGreater(windows.features.shape[0], 0)

    def test_window_mask_right_aligned(self) -> None:
        episode = load_episode(self.files[0])
        windows = build_windows(episode, prefixes_s=PREFIXES, training_hz=50.0, stride=2)
        # 0.10 s 前缀在 50 Hz 下是 6 帧；其余位置应被 mask 屏蔽。
        short = np.isclose(windows.prefix_s, 0.10)
        self.assertTrue(short.any())
        counts = windows.mask[short].sum(axis=1)
        self.assertTrue(np.all(counts <= 6))
        # mask 必须右对齐：最后一位恒为 True。
        self.assertTrue(windows.mask[short][:, -1].all())

    def test_labels_and_validity(self) -> None:
        episode = load_episode(self.files[0])
        windows = build_windows(episode, prefixes_s=PREFIXES, training_hz=50.0, stride=2)
        # 接触期内都是纯接触 -> 有效性全为 1。
        self.assertTrue(windows.contact_valid.all())
        np.testing.assert_allclose(windows.temperature, 35.0, atol=1e-6)

    def test_failed_episode_has_negative_windows(self) -> None:
        episode = load_episode(self.failed)
        windows = build_windows(episode, prefixes_s=PREFIXES, training_hz=50.0, stride=2)
        self.assertGreater(windows.features.shape[0], 0)
        self.assertFalse(windows.contact_valid.any())
        self.assertTrue(np.all(np.isnan(windows.temperature)))

    def test_no_contact_window_times(self) -> None:
        episode = load_episode(self.failed)
        ends = window_end_times(
            episode.arrays,
            min_contact_s=0.05,
            window_stride_s=0.05,
            no_contact_window_s=0.0,
        )
        self.assertEqual(ends.size, 0)
        ends = window_end_times(
            episode.arrays,
            min_contact_s=0.05,
            window_stride_s=0.05,
            no_contact_window_s=0.5,
        )
        self.assertGreaterEqual(ends.size, 9)

    def test_dataset_and_stats(self) -> None:
        stats = stats_stream(
            [self.files[0]], prefixes_s=PREFIXES, training_hz=50.0, stride=2
        )
        self.assertEqual(len(stats["mean"]), len(ENCODER_INPUT_CHANNELS))
        dataset = ThermalWindowDataset(
            [self.files[0], self.failed],
            prefixes_s=PREFIXES,
            training_hz=50.0,
            stride=2,
            stats=stats,
        )
        total = sum(int(dataset.counts[i]) for i in range(len(dataset.files)))
        self.assertEqual(len(dataset), total)
        sample = dataset[0]
        self.assertEqual(tuple(sample["features"].shape), (26, len(ENCODER_INPUT_CHANNELS)))
        self.assertEqual(tuple(sample["mask"].shape), (26,))
        # 归一化后 mask 之外的帧必须为 0。
        off = ~sample["mask"].numpy()
        self.assertTrue(np.all(sample["features"].numpy()[off] == 0.0))

    def test_compute_stats_from_windows(self) -> None:
        windows = build_windows(load_episode(self.files[0]), prefixes_s=PREFIXES, training_hz=50.0, stride=2)
        stats = compute_stats(windows)
        self.assertEqual(len(stats["mean"]), len(ENCODER_INPUT_CHANNELS))
        self.assertTrue(all(v > 0 for v in stats["std"]))


if __name__ == "__main__":
    unittest.main()

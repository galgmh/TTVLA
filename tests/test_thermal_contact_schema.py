"""数据契约与记录器测试（不需要仿真，numpy + h5py 即可）。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.schema.thermal_contact_v1 import (  # noqa: E402
    ENCODER_INPUT_CHANNELS,
    FRAME_FIELDS,
    PHASES,
    empty_episode,
    metadata_for_spec,
    stack_episode,
    validate_arrays,
)
from data.writer.hdf5_writer import read_episode, write_episode  # noqa: E402
from data.writer.recorder import EpisodeRecorder  # noqa: E402
from sim.sampling.episode_spec import EpisodeSpec  # noqa: E402


def _frame(phase: int = 0, temp: float = 25.0) -> dict:
    return {
        "time/sim_time": 0.0,
        "obs/fingertip_temp": np.array([temp, temp - 0.6]),
        "obs/contact_force_world": np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
        "obs/contact_normal_force": np.array([1.0, 1.0]),
        "obs/contact_mask": np.array([True, True]),
        "obs/gripper_width": 0.06,
        "obs/finger_joint_position": np.array([0.03, 0.03]),
        "obs/eef_pose": np.array([0.3, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]),
        "obs/contact_area": np.array([1e-4, 1e-4]),
        "action/eef_target": np.array([0.3, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]),
        "action/gripper_target": 0.06,
        "action/target_force": 2.0,
        "action/probe_region": 1,
        "action/phase": phase,
        "privileged/link_temp": np.array([20.0, 35.0, 50.0, 25.0]),
        "privileged/contact_link_mask": np.array([[False, True, False, False], [False, True, False, False]]),
        "privileged/pure_contact": True,
        "privileged/contact_valid": True,
    }


def _spec(**overrides) -> EpisodeSpec:
    base = dict(
        episode_id="test_000000",
        dataset_version="thermalcontact-v1",
        split="pilot",
        random_seed=7,
        region_order=("lower", "middle", "upper", "handle"),
        target_region=1,
        mixed=False,
        regime="case_1",
        finger_mode="ambient",
        link_temps_c=(20.0, 35.0, 50.0, 25.0),
        fingertip_temps_c=(24.0, 23.0),
        force_level=1.0,
        target_force_n=2.0,
        contact_quality="stable",
        contact_hold_s=0.5,
        fail_retreat_m=0.0,
        sensor_rc_s=0.05,
        sensor_noise_c=0.05,
        sensor_bias_c=0.0,
        sensor_quantization_c=0.1,
        probe_pos=(0.0, 0.0, 0.055),
        probe_tool_dir=(-1.0, 0.0, 0.0),
        probe_finger_dir=(0.0, -1.0, 0.0),
        probe_pre_dist=0.06,
        cup_xy_jitter_m=(0.0, 0.0),
        cup_yaw_jitter_rad=0.0,
        material_parameters=(("conductivity", 1.5), ("density", 2300.0)),
    )
    base.update(overrides)
    return EpisodeSpec(**base)


class SchemaTest(unittest.TestCase):
    def test_channel_count(self) -> None:
        self.assertEqual(len(ENCODER_INPUT_CHANNELS), 14)

    def test_stack_roundtrip(self) -> None:
        buffers = empty_episode()
        for i in range(5):
            frame = _frame(phase=min(i, len(PHASES) - 1))
            frame["time/sim_time"] = 0.01 * i
            for key, value in frame.items():
                buffers[key].append(value)
        arrays = stack_episode(buffers)
        for name, spec in FRAME_FIELDS.items():
            self.assertEqual(arrays[name].shape, (5, *spec.shape), name)
        self.assertEqual(validate_arrays(arrays), [])

    def test_validate_rejects_bad_data(self) -> None:
        buffers = empty_episode()
        for i in range(3):
            frame = _frame()
            frame["time/sim_time"] = 0.01 * i
            for key, value in frame.items():
                buffers[key].append(value)
        arrays = stack_episode(buffers)
        bad = dict(arrays)
        bad["time/sim_time"] = np.array([0.0, 0.01, 0.01])
        self.assertTrue(any("严格递增" in msg for msg in validate_arrays(bad)))

        bad = dict(arrays)
        bad["obs/contact_normal_force"] = np.array([[1.0, -2.0]] * 3)
        self.assertTrue(any("负值" in msg for msg in validate_arrays(bad)))

        bad = dict(arrays)
        bad["action/phase"] = np.array([0, 1, 99])
        self.assertTrue(any("阶段码" in msg for msg in validate_arrays(bad)))

    def test_hdf5_roundtrip(self) -> None:
        buffers = empty_episode()
        for i in range(6):
            frame = _frame(phase=2)
            frame["time/sim_time"] = 0.01 * i
            for key, value in frame.items():
                buffers[key].append(value)
        spec = _spec()

        class Outcome:
            contact_start_s = 0.02
            contact_end_s = 0.05
            termination_reason = "contact_stable"
            policy_steps = 6
            peak_normal_force_n = 1.9

        metadata = metadata_for_spec(spec, cup_pose=np.arange(7.0), cup_asset_version="cup4link-v1", outcome=Outcome())
        scalars = {"privileged/target_region": 1}
        with tempfile.TemporaryDirectory() as tmp:
            path = write_episode(Path(tmp) / "ep.hdf5", buffers, metadata, scalars)
            arrays, meta, sc = read_episode(path)
        self.assertEqual(meta["episode_id"], "test_000000")
        self.assertEqual(meta["format_version"], "thermalcontact-v1.8")
        np.testing.assert_allclose(arrays["obs/fingertip_temp"][0], [25.0, 24.4])
        self.assertEqual(int(sc["privileged/target_region"]), 1)


class RecorderTest(unittest.TestCase):
    def _privileged(self, mask, foreign=False):
        return {
            "link_temp": np.array([20.0, 35.0, 50.0, 25.0]),
            "contact_link_mask": np.asarray(mask, dtype=bool),
            "contact_foreign": np.asarray(foreign),
        }

    def _policy(self):
        return {
            "fingertip_temp": np.array([25.0, 24.0]),
            "contact_force_world": np.zeros((2, 3)),
            "contact_normal_force": np.zeros(2),
            "contact_mask": np.array([False, False]),
            "gripper_width": 0.07,
            "finger_joint_position": np.array([0.035, 0.035]),
            "eef_pose": np.zeros(7),
            "contact_area": np.zeros(2),
        }

    def test_pure_contact_only_for_target_region(self) -> None:
        recorder = EpisodeRecorder(target_region=1)
        command = (np.zeros(7), 0.07)
        mask_ok = np.zeros((2, 4), dtype=bool)
        mask_ok[0, 1] = True
        recorder.add(
            sim_time=0.0,
            policy=self._policy(),
            privileged=self._privileged(mask_ok),
            command=command,
            target_force_n=2.0,
            probe_region=1,
            phase="HOLD",
        )
        self.assertTrue(recorder.buffers["privileged/pure_contact"][-1])
        self.assertTrue(recorder.buffers["privileged/contact_valid"][-1])

        # 碰到两个区域 -> 不是纯接触，但接触仍然有效。
        mask_two = np.zeros((2, 4), dtype=bool)
        mask_two[0, 1] = True
        mask_two[1, 2] = True
        recorder.add(
            sim_time=0.01,
            policy=self._policy(),
            privileged=self._privileged(mask_two),
            command=command,
            target_force_n=2.0,
            probe_region=1,
            phase="HOLD",
        )
        self.assertFalse(recorder.buffers["privileged/pure_contact"][-1])
        self.assertTrue(recorder.buffers["privileged/contact_valid"][-1])

        # 外来 link（手）碰到杯子 -> 不纯，但仍算有效接触。
        recorder.add(
            sim_time=0.02,
            policy=self._policy(),
            privileged=self._privileged(mask_ok, foreign=True),
            command=command,
            target_force_n=2.0,
            probe_region=1,
            phase="HOLD",
        )
        self.assertFalse(recorder.buffers["privileged/pure_contact"][-1])

        # 没有接触。
        recorder.add(
            sim_time=0.03,
            policy=self._policy(),
            privileged=self._privileged(np.zeros((2, 4), dtype=bool)),
            command=command,
            target_force_n=2.0,
            probe_region=1,
            phase="SEEK_CONTACT",
        )
        self.assertFalse(recorder.buffers["privileged/pure_contact"][-1])
        self.assertFalse(recorder.buffers["privileged/contact_valid"][-1])


if __name__ == "__main__":
    unittest.main()

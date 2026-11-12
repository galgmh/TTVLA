"""多时间轴 HDF5 契约、记录器与读写回归测试。"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from data.schema.thermal_contact_v1 import ENCODER_INPUT_CHANNELS, FORMAT_VERSION, validate_arrays
from data.writer.hdf5_writer import read_episode, write_episode
from data.writer.recorder import EpisodeRecorder
from sim.sampling.episode_spec import EpisodeSpec


def _spec() -> EpisodeSpec:
    return EpisodeSpec(
        episode_id="test_000000", dataset_version="thermalcontact-v1", split="pilot",
        random_seed=7, region_order=("lower", "middle", "upper", "handle"),
        target_region=1, mixed=False, regime="case_1", finger_mode="ambient",
        link_temps_c=(20.0, 35.0, 50.0, 25.0), fingertip_temps_c=(24.0, 23.0),
        force_level=1.0, target_force_n=2.0, contact_quality="stable",
        contact_hold_s=0.5, fail_retreat_m=0.0, sensor_rc_s=0.05,
        sensor_noise_c=0.05, sensor_bias_c=0.0, sensor_quantization_c=0.1,
        probe_pos=(0.0, 0.0, 0.055), probe_tool_dir=(-1.0, 0.0, 0.0),
        probe_finger_dir=(0.0, -1.0, 0.0), probe_pre_dist=0.06,
        cup_xy_jitter_m=(0.0, 0.0), cup_yaw_jitter_rad=0.0,
        material_parameters=(("conductivity", 1.5), ("density", 2300.0)),
    )


def _tactile(contact: bool) -> dict[str, np.ndarray]:
    return {
        "fingertip_temp": np.array([25.0, 24.0]),
        "contact_force_link": np.array([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]]) * contact,
        "contact_normal_force": np.array([1.0, 1.0]) * contact,
        "contact_area": np.array([1e-4, 1e-4]) * contact,
        "contact_mask": np.array([contact, contact]),
        "sensor_position_base": np.array([[0.4, -0.03, 0.5], [0.4, 0.03, 0.5]]),
        "sensor_normal_base": np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]),
    }


def _privileged(contact: bool) -> dict[str, np.ndarray]:
    mask = np.zeros((2, 4), dtype=bool)
    if contact:
        mask[:, 1] = True
    return {"link_temp": np.array([20.0, 35.0, 50.0, 25.0]), "contact_link_mask": mask}


def _robot() -> dict[str, np.ndarray]:
    return {
        "eef_pose": np.array([0.3, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]),
        "joint_position": np.zeros(9), "joint_velocity": np.zeros(9), "gripper_width": 0.06,
    }


def _camera() -> dict[str, np.ndarray]:
    return {
        "base_0_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
        "wrist_0_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
        "wrist_0_pose": np.array([0.3, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]),
    }


def _calibration() -> dict[str, np.ndarray]:
    return {
        "base_0_intrinsics": np.eye(3), "wrist_0_intrinsics": np.eye(3),
        "base_0_pose": np.array([0.8, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]),
        "hand_to_wrist_camera": np.eye(4),
    }


def _episode() -> EpisodeRecorder:
    recorder = EpisodeRecorder(target_region=1)
    recorder.set_camera_calibration(_calibration())
    # 接触区间 [2,5)。
    for i in range(6):
        contact = 2 <= i < 5
        sensor_index = recorder.add_sensor(
            sim_time=0.01 * i, tactile=_tactile(contact), privileged=_privileged(contact)
        )
        if i in (0, 5):
            recorder.add_control(
                sim_time=0.01 * i, sensor_index=sensor_index,
                command=(np.array([0.3, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]), 0.06),
                target_force_n=2.0, robot=_robot(), camera=_camera(),
            )
    recorder.finalize()
    return recorder


def test_channel_count() -> None:
    assert len(ENCODER_INPUT_CHANNELS) == 14


def test_event_boundaries_and_per_finger_validity() -> None:
    recorder = _episode()
    assert recorder.buffers["contact_events/start_sensor_index"] == [2]
    assert recorder.buffers["contact_events/end_sensor_index_exclusive"] == [5]
    np.testing.assert_array_equal(recorder.buffers["tactile/contact_event_id"], [0, 0, 1, 1, 1, 0])
    np.testing.assert_array_equal(recorder.buffers["privileged/contact_valid"][2], [True, True])


def test_hdf5_roundtrip() -> None:
    recorder = _episode()
    spec = _spec()

    class Outcome:
        termination_reason = "contact_stable"
        policy_steps = 2
        peak_normal_force_n = 1.9

    from data.schema.thermal_contact_v1 import metadata_for_spec

    metadata = metadata_for_spec(
        spec, genesis_version="1.4.1", cup_pose=np.arange(7.0),
        cup_asset_version="cup4link-v1", outcome=Outcome(),
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = write_episode(Path(tmp) / "ep.hdf5", recorder.buffers, metadata, recorder.static)
        arrays, loaded_metadata, static = read_episode(path)
    assert loaded_metadata["format_version"] == FORMAT_VERSION
    assert loaded_metadata["episode_id"] == "test_000000"
    assert validate_arrays(arrays) == []
    assert arrays["tactile/fingertip_temp"].shape == (6, 2)
    assert arrays["camera/base_0_rgb"].shape == (2, 8, 8, 3)
    assert static["camera/hand_to_wrist_camera"].shape == (4, 4)


def test_validation_rejects_future_sensor_index() -> None:
    recorder = _episode()
    from data.schema.thermal_contact_v1 import stack_episode

    arrays = {**stack_episode(recorder.buffers), **recorder.static}
    arrays["control/sensor_index"] = np.array([0, 99])
    assert any("sensor_index" in issue for issue in validate_arrays(arrays))

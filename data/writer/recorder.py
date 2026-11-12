"""100 Hz 传感器与 20 Hz 控制/RGB 多时间轴 episode 记录器。"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from data.schema.thermal_contact_v1 import empty_episode


class EpisodeRecorder:
    """记录一集原始数据，并在有效接触的上升/下降沿生成事件。"""

    def __init__(self, *, target_region: int, link_count: int = 4, event_off_frames: int = 5):
        self.target_region = int(target_region)
        self.link_count = int(link_count)
        self.buffers: dict[str, list] = empty_episode()
        self.static: dict[str, np.ndarray] = {}
        self.sensor_frames = 0
        self.control_frames = 0
        self._active_event_id = 0
        self._active_event_start: int | None = None
        self._event_off_frames = int(event_off_frames)
        self._gap_start: int | None = None

    def add_sensor(
        self,
        *,
        sim_time: float,
        tactile: Mapping[str, np.ndarray],
        privileged: Mapping[str, np.ndarray],
    ) -> int:
        """追加一帧 100 Hz 触觉/热觉数据，返回新帧索引。"""
        link_mask = np.asarray(privileged["contact_link_mask"], dtype=bool).reshape(2, self.link_count)
        contact_valid = link_mask.any(axis=1)
        any_valid = bool(contact_valid.any())
        index = self.sensor_frames

        if any_valid:
            if self._active_event_start is None:
                self._active_event_id += 1
                self._active_event_start = index
                self.buffers["contact_events/start_sensor_index"].append(index)
            self._gap_start = None
            event_id = self._active_event_id
        elif self._active_event_start is not None:
            if self._gap_start is None:
                self._gap_start = index
            gap_length = index - self._gap_start + 1
            if gap_length < self._event_off_frames:
                # 短暂断触先归入当前事件；超过滞回阈值时再回溯改为非接触。
                event_id = self._active_event_id
            else:
                for prior in range(self._gap_start, index):
                    self.buffers["tactile/contact_event_id"][prior] = 0
                self.buffers["contact_events/end_sensor_index_exclusive"].append(self._gap_start)
                self._active_event_start = None
                self._gap_start = None
                event_id = 0
        else:
            event_id = 0
        b = self.buffers
        b["sensor/time"].append(float(sim_time))
        b["tactile/fingertip_temp"].append(np.asarray(tactile["fingertip_temp"], dtype=np.float32))
        b["tactile/contact_force_link"].append(np.asarray(tactile["contact_force_link"], dtype=np.float32))
        b["tactile/contact_normal_force"].append(np.asarray(tactile["contact_normal_force"], dtype=np.float32))
        b["tactile/contact_area"].append(np.asarray(tactile["contact_area"], dtype=np.float32))
        b["tactile/contact_mask"].append(np.asarray(tactile["contact_mask"], dtype=bool))
        b["tactile/contact_event_id"].append(int(event_id))
        b["tactile/sensor_position_base"].append(np.asarray(tactile["sensor_position_base"], dtype=np.float32))
        b["tactile/sensor_normal_base"].append(np.asarray(tactile["sensor_normal_base"], dtype=np.float32))
        b["privileged/link_temp"].append(np.asarray(privileged["link_temp"], dtype=np.float32))
        b["privileged/contact_link_mask"].append(link_mask)
        b["privileged/contact_valid"].append(contact_valid)
        self.sensor_frames += 1
        return index

    def add_control(
        self,
        *,
        sim_time: float,
        sensor_index: int,
        command: tuple[np.ndarray, float],
        target_force_n: float,
        robot: Mapping[str, np.ndarray],
        camera: Mapping[str, np.ndarray],
    ) -> None:
        """追加一帧 20 Hz 控制、机器人状态与 RGB。"""
        if not 0 <= int(sensor_index) < self.sensor_frames:
            raise ValueError(f"sensor_index={sensor_index} 超出已记录的 {self.sensor_frames} 帧")
        eef_target, gripper_target = command
        b = self.buffers
        b["control/time"].append(float(sim_time))
        b["control/sensor_index"].append(int(sensor_index))
        b["control/eef_target"].append(np.asarray(eef_target, dtype=np.float32))
        b["control/gripper_target"].append(float(gripper_target))
        b["control/target_force"].append(float(target_force_n))
        b["robot/eef_pose"].append(np.asarray(robot["eef_pose"], dtype=np.float32))
        b["robot/joint_position"].append(np.asarray(robot["joint_position"], dtype=np.float32))
        b["robot/joint_velocity"].append(np.asarray(robot["joint_velocity"], dtype=np.float32))
        b["robot/gripper_width"].append(float(robot["gripper_width"]))
        b["camera/base_0_rgb"].append(np.asarray(camera["base_0_rgb"], dtype=np.uint8))
        b["camera/wrist_0_rgb"].append(np.asarray(camera["wrist_0_rgb"], dtype=np.uint8))
        b["camera/wrist_0_pose"].append(np.asarray(camera["wrist_0_pose"], dtype=np.float32))
        self.control_frames += 1

    def set_camera_calibration(self, calibration: Mapping[str, np.ndarray]) -> None:
        """保存每集静态的相机内外参。"""
        for key in (
            "base_0_intrinsics", "wrist_0_intrinsics", "base_0_pose",
            "hand_to_wrist_camera",
        ):
            if key not in calibration:
                raise KeyError(f"相机标定缺少 {key}")
            self.static[f"camera/{key}"] = np.asarray(calibration[key], dtype=np.float32)

    def finalize(self) -> None:
        """关闭 episode 末尾仍在进行的接触事件。"""
        if self._active_event_start is not None:
            end = self.sensor_frames if self._gap_start is None else self._gap_start
            if self._gap_start is not None:
                for prior in range(self._gap_start, self.sensor_frames):
                    self.buffers["tactile/contact_event_id"][prior] = 0
            self.buffers["contact_events/end_sensor_index_exclusive"].append(end)
            self._active_event_start = None
            self._gap_start = None

    def summary(self) -> dict[str, Any]:
        valid = np.asarray(self.buffers["privileged/contact_valid"], dtype=bool)
        return {
            "sensor_frames": self.sensor_frames,
            "control_frames": self.control_frames,
            "contact_events": len(self.buffers["contact_events/start_sensor_index"]),
            "valid_contact_samples_per_finger": valid.sum(axis=0).tolist() if valid.size else [0, 0],
        }


__all__ = ["EpisodeRecorder"]

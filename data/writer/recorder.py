"""逐帧记录器：把传感器读数与指令填进 HDF5 契约的缓冲。

记录器本身不读仿真——生成脚本每个采样时刻把观测字典与指令传给
:meth:`EpisodeRecorder.add`。这样记录逻辑可以脱离引擎单测，也不会在
记录层偷偷做派生计算。

``pure_contact`` / ``contact_valid`` 在这里由接触归属表派生：

* ``contact_valid`` = 任一手指与杯子接触；
* ``pure_contact``  = 接触有效、没有“外来”机器人 link 接触、且只有 *目标*
  区域被碰到。混杂接触集（``target_region < 0``）永远为假。
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from data.schema.thermal_contact_v1 import PHASE_CODES, empty_episode


class EpisodeRecorder:
    """一集的逐帧缓冲。"""

    def __init__(self, *, target_region: int):
        self.target_region = int(target_region)
        self.buffers: dict[str, list] = empty_episode()
        self.frames = 0

    def add(
        self,
        *,
        sim_time: float,
        policy: Mapping[str, np.ndarray],
        privileged: Mapping[str, np.ndarray],
        command: tuple[np.ndarray, float],
        target_force_n: float,
        probe_region: int,
        phase: str,
    ) -> None:
        """追加一帧。``policy`` / ``privileged`` 来自 ``sim.obs`` 的两个观测函数。"""
        eef_target, gripper_target = command
        link_mask = np.asarray(privileged["contact_link_mask"], dtype=bool).reshape(2, 4)
        foreign = bool(np.asarray(privileged["contact_foreign"]).reshape(()))

        touched = np.where(link_mask.any(axis=0))[0]
        valid = bool(touched.size > 0)
        pure = bool(
            valid
            and not foreign
            and self.target_region >= 0
            and touched.size == 1
            and int(touched[0]) == self.target_region
        )

        b = self.buffers
        b["time/sim_time"].append(float(sim_time))
        b["obs/fingertip_temp"].append(np.asarray(policy["fingertip_temp"], dtype=np.float64))
        b["obs/contact_force_world"].append(np.asarray(policy["contact_force_world"], dtype=np.float64))
        b["obs/contact_normal_force"].append(np.asarray(policy["contact_normal_force"], dtype=np.float64))
        b["obs/contact_mask"].append(np.asarray(policy["contact_mask"], dtype=bool))
        b["obs/gripper_width"].append(float(policy["gripper_width"]))
        b["obs/finger_joint_position"].append(np.asarray(policy["finger_joint_position"], dtype=np.float64))
        b["obs/eef_pose"].append(np.asarray(policy["eef_pose"], dtype=np.float64))
        b["obs/contact_area"].append(np.asarray(policy["contact_area"], dtype=np.float64))
        b["action/eef_target"].append(np.asarray(eef_target, dtype=np.float64))
        b["action/gripper_target"].append(float(gripper_target))
        b["action/target_force"].append(float(target_force_n))
        b["action/probe_region"].append(int(probe_region))
        b["action/phase"].append(int(PHASE_CODES[phase]))
        b["privileged/link_temp"].append(np.asarray(privileged["link_temp"], dtype=np.float64))
        b["privileged/contact_link_mask"].append(link_mask)
        b["privileged/pure_contact"].append(pure)
        b["privileged/contact_valid"].append(valid)
        self.frames += 1

    def summary(self) -> dict[str, Any]:
        """帧层面的小统计（写进元数据，便于事后审计）。"""
        if not self.buffers["time/sim_time"]:
            return {"frames": 0}
        return {
            "frames": self.frames,
            "pure_contact_frames": int(np.sum(self.buffers["privileged/pure_contact"])),
            "contact_frames": int(np.sum(self.buffers["privileged/contact_valid"])),
        }


__all__ = ["EpisodeRecorder"]

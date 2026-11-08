"""观测读取：策略观测（模型输入）与特权观测（标签/审计用）。

两类观测严格分开：

* ``get_policy_observation`` 只读指尖温度、指尖接触力、夹爪开口、指位，
  以及由凝胶标记场估计的接触面积——这是编码器唯一允许看到的信号；
* ``get_privileged_observation`` 读真实 link 温度与“哪根手指碰到哪个热区”
  的接触归属，只进数据文件，绝不进模型输入。

``contact_mask`` 是逐帧的真实接触判定（接触力传感器或法向力超过阈值）；
``contact_link_mask`` 是 [2,4] 的“手指-热区”布尔表，用于校验选集纯度。
"""

from __future__ import annotations

import numpy as np

from sim.control.actions import tcp_pose
from sim.rotations import quat_rotate
from sim.sensors.tactile import read_all
from sim.thermal.link_temperature import read_region_temperatures

# 判定“有接触”的法向力阈值（N）。
CONTACT_FORCE_EPS_N = 0.05
# 凝胶标记被压缩到这个法向位移（m）才算接触点。实测（2.0 N 级接触）：
# 无接触时标记位移严格为 0，接触后最大压缩约 13 µm；取 5 µm（约峰值的
# 40%）让面积随接触力单调变化，又不至于把整个垫面都算成接触。
CONTACT_DISPLACEMENT_M = 5.0e-6


def _numpy(tensor) -> np.ndarray:
    return np.asarray(tensor.detach().cpu(), dtype=np.float64)


def finger_poses(bundle) -> tuple[np.ndarray, np.ndarray]:
    """两根手指 link 的世界位姿 (pos (2,3), wxyz quat (2,4))。"""
    pos = []
    quat = []
    for name in bundle.finger_link_names:
        link = bundle.robot.get_link(name)
        pos.append(_numpy(link.get_pos()).reshape(-1)[:3])
        quat.append(_numpy(link.get_quat()).reshape(-1)[:4])
    return np.stack(pos), np.stack(quat)


def _elastomer_handles(bundle) -> list:
    return [h for h in bundle.tactile if h.modality == "elastomer"]


def contact_area(bundle) -> np.ndarray:
    """由凝胶标记场估计的接触面积 (2,)，单位 m²。

    每个标记的位移投影到“指向另一根手指”的垫面法向；法向压缩量超过
    ``CONTACT_DISPLACEMENT_M`` 的标记判定为接触点，面积 = 接触点数 /
    总点数 × 垫面面积。这是模型可用的面积信号（不是真实接触几何）。
    """
    handles = _elastomer_handles(bundle)
    if not handles:
        return np.zeros(2, dtype=np.float64)
    data = read_all(handles, temporal_reduction="median")
    areas = np.zeros(2, dtype=np.float64)
    for i, handle in enumerate(handles):
        disp = _numpy(data[handle.name])
        disp = disp.reshape(-1, 3)
        layout = bundle.tactile_layouts["elastomer"][i]
        normal = np.asarray(layout.inner_dir, dtype=np.float64)
        compression = np.abs(disp @ normal)
        hit = float(np.mean(compression > CONTACT_DISPLACEMENT_M))
        areas[i] = hit * float(layout.pad_face_area_m2)
    return areas


def get_policy_observation(bundle) -> dict[str, np.ndarray]:
    """策略观测（编码器输入的唯一来源）。"""
    from sim.sensors.temperature import read_temperature

    temp_data = read_temperature(bundle.temperatures)
    # 指尖网格是 (1,1,1)：每个 handle 取唯一的测量值。
    temp = np.asarray(
        [_numpy(temp_data[h.name]).reshape(-1)[0] for h in bundle.temperatures], dtype=np.float64
    )

    data = read_all(bundle.link_contacts, temporal_reduction="last")
    normal = np.zeros(2, dtype=np.float64)
    touching = np.zeros(2, dtype=bool)
    force_link = np.zeros((2, 3), dtype=np.float64)
    inner = [np.asarray(lay.inner_dir, dtype=np.float64) for lay in bundle.layouts]
    for i, name in enumerate(bundle.finger_link_names):
        fname = f"tactile_link_force_{name}"
        if fname in data:
            force_link[i] = _numpy(data[fname]).reshape(3)
        normal[i] = max(0.0, -float(force_link[i] @ inner[i]))
        bname = f"tactile_link_bool_{name}"
        if bname in data:
            touching[i] = bool(_numpy(data[bname]).reshape(-1)[0] > 0.5)

    _, quat = finger_poses(bundle)
    force_world = np.stack([quat_rotate(quat[i], force_link[i]) for i in range(2)])
    eef_pos, eef_quat = tcp_pose(bundle)

    return {
        "fingertip_temp": temp,
        "contact_force_world": force_world,
        "contact_normal_force": normal,
        "contact_mask": (touching | (normal > CONTACT_FORCE_EPS_N)).astype(bool),
        "gripper_width": np.asarray(bundle.gripper_width(), dtype=np.float64),
        "finger_joint_position": bundle.finger_positions(),
        "eef_pose": np.concatenate([eef_pos, eef_quat]),
        "contact_area": contact_area(bundle),
    }


def contact_link_mask(bundle) -> tuple[np.ndarray, bool]:
    """手指-热区接触归属 (2,4) 与“是否只有一根手指碰到杯子其它部位”。

    返回 ``(mask, foreign)``：``mask[f, r]`` 表示手指 f 与热区 r 的 link 有
    接触；``foreign`` 表示机器人上除两根手指以外的 link（手、腕、臂）碰到了
    杯子——这种集不能当纯接触样本。
    """
    mask = np.zeros((2, 4), dtype=bool)
    foreign = False
    contacts = bundle.cup.get_contacts()
    if not contacts:
        return mask, foreign

    def _flatten(key: str) -> np.ndarray:
        if key not in contacts:
            return np.zeros(0, dtype=np.int64)
        return _numpy(contacts[key]).reshape(-1)

    link_a = _flatten("link_a").astype(np.int64)
    link_b = _flatten("link_b").astype(np.int64)
    if link_a.size != link_b.size:
        raise ValueError(f"接触表的 link_a/link_b 长度不一致：{link_a.size} vs {link_b.size}")
    finger_idx = np.asarray([int(bundle.robot.get_link(n).idx) for n in bundle.finger_link_names])
    robot_idx = {int(link.idx) for link in bundle.robot.links}

    for a, b in zip(link_a, link_b):
        region = bundle.regions.region_of_link(int(a))
        other = int(b)
        if region < 0:
            region = bundle.regions.region_of_link(int(b))
            other = int(a)
        if region < 0:
            continue
        if other in robot_idx:
            hits = np.where(finger_idx == other)[0]
            if hits.size == 1:
                mask[int(hits[0]), region] = True
            else:
                foreign = True
        # 平台/桌面等非机器人接触不参与纯接触判定。
    return mask, foreign


def get_privileged_observation(bundle) -> dict[str, np.ndarray]:
    """特权观测：真实 link 温度与接触归属（不进入模型输入）。"""
    link_temp = np.asarray(
        read_region_temperatures(bundle.temperatures[0], bundle.regions).detach().cpu(),
        dtype=np.float64,
    ).reshape(-1)
    mask, foreign = contact_link_mask(bundle)
    return {
        "link_temp": link_temp,
        "contact_link_mask": mask,
        "contact_foreign": np.asarray(foreign),
    }


__all__ = [
    "CONTACT_DISPLACEMENT_M",
    "CONTACT_FORCE_EPS_N",
    "contact_area",
    "contact_link_mask",
    "finger_poses",
    "get_policy_observation",
    "get_privileged_observation",
]

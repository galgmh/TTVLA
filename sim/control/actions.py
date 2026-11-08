"""TCP 位姿控制：世界系目标 -> 7 个臂关节 + 2 个指关节的位置指令。

两层：

* ``point_world`` 把“杯子作者坐标系里的探测位姿”换算成世界系 TCP 目标。
  探测点的 z 轴是接近方向 ``tool_dir``，y 轴是两指闭合方向 ``finger_dir``，
  x 轴取 ``y × z``；这点决定了指尖垫面覆盖杯子哪一段热区。
* ``DeltaPoseController`` 每个策略步把*指令位姿*朝目标位姿推进有限的一步
  （位置限速 + 姿态限速），再用阻尼最小二乘 IK 解出关节指令，指关节按
  限速逼近目标开口。限速是必须的：IK 直接吃远处目标会跳到另一个解支
  （例如腕部拧转 90°），关节指令突变会让手臂甩出去。IK 只用当前状态作
  初值（``max_samples=1``）并从实测 q 重锚，因此策略步之间不会累积位姿
  漂移。

手指开合不做位置伺服之外的力控制：力闭环在 ``probe_controller`` 里，通过
修正指位目标实现（指 PD 刚度把“多压多少”变成“多大法向力”）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.rotations import (
    quat_apply_delta,
    quat_error_rotvec,
    quat_mul,
    quat_normalize,
    quat_rotate,
)

# 每个策略步允许的最大关节增量（rad）：目标突变时不会甩飞，也保护 IK 的
# 线性化假设。
MAX_JOINT_STEP_RAD = 0.25


@dataclass(frozen=True)
class ProbePose:
    """一次探测的三个世界系位姿：预接近位、接触位、接近方向。"""

    pos: np.ndarray  # (3,) TCP 接触目标
    quat: np.ndarray  # (4,) wxyz 手 link 朝向
    pre_pos: np.ndarray  # (3,) 沿接近方向退回 pre_dist 的预接触位
    approach_dir: np.ndarray  # (3,) 世界系单位接近方向


def _unit(vec, where: str) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise ValueError(f"{where} 不能是零向量")
    return v / n


def point_world(
    obj_pos,
    obj_quat,
    *,
    pos_local=(0.0, 0.0, 0.0),
    tool_dir=(0.0, 0.0, -1.0),
    finger_dir=(0.0, -1.0, 0.0),
    pre_dist: float = 0.0,
) -> ProbePose:
    """杯子作者坐标系里的探测位姿 -> 世界系 ``ProbePose``。

    ``pos_local`` 是 TCP 在杯子作者坐标系里的位置；``tool_dir`` 是手 z 轴
    （手指指向），``finger_dir`` 是手 y 轴（闭合方向）。两者在作者坐标系里
    给定，整体随杯子位姿旋转。``pre_dist`` 沿接近方向的反方向退回。
    """
    from scipy.spatial.transform import Rotation

    obj_pos = np.asarray(obj_pos, dtype=np.float64).reshape(3)
    obj_quat = quat_normalize(obj_quat)

    z_axis = _unit(tool_dir, "tool_dir")
    y_axis = np.asarray(finger_dir, dtype=np.float64).reshape(3)
    y_axis = y_axis - float(y_axis @ z_axis) * z_axis  # 正交化
    y_axis = _unit(y_axis, "finger_dir（与 tool_dir 正交化后）")
    x_axis = np.cross(y_axis, z_axis)

    rot_local = np.column_stack([x_axis, y_axis, z_axis])
    xyzw = Rotation.from_matrix(rot_local).as_quat()
    q_local = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)

    pos = obj_pos + quat_rotate(obj_quat, pos_local)
    quat = quat_normalize(quat_mul(obj_quat, q_local))
    approach = quat_rotate(obj_quat, z_axis)
    pre_pos = pos - float(pre_dist) * approach
    return ProbePose(pos=pos, quat=quat, pre_pos=pre_pos, approach_dir=approach)


def tcp_pose(bundle) -> tuple[np.ndarray, np.ndarray]:
    """当前 TCP 位姿：手 link 位姿 + ``tcp_offset``（手坐标系）。"""
    pos, quat = bundle.hand_pose()
    offset = np.asarray(bundle.cfg["robot"]["tcp_offset"], dtype=np.float64)
    return pos + quat_rotate(quat, offset), quat


class DeltaPoseController:
    """限速位姿控制器：每个策略步推进指令位姿一次，解一次 IK。"""

    def __init__(self, bundle):
        self.bundle = bundle
        ctrl = bundle.cfg["controller"]
        robot = bundle.cfg["robot"]
        self.arm_dofs = np.asarray(bundle.arm_dofs, dtype=np.int64)
        self.finger_dofs = np.asarray(bundle.finger_dofs, dtype=np.int64)
        self.tcp_offset = np.asarray(robot["tcp_offset"], dtype=np.float64)
        self.damping = float(ctrl["ik_damping"])
        self.max_iters = int(ctrl["ik_max_iters"])
        self.max_samples = int(ctrl["ik_max_samples"])
        self.ik_step = float(ctrl["ik_max_step_rad"])
        self.dt = float(ctrl["decimation"]) * float(ctrl["sim_dt"])
        self.width_min = 2.0 * float(robot["closed_finger_pos"])
        self.width_max = 2.0 * float(robot["open_finger_pos"])
        # 指令位姿的推进限速（每策略步）。
        self.max_pos_step = float(ctrl["pos_speed_m_per_s"]) * self.dt
        self.max_rot_step = float(ctrl["rot_speed_rad_per_s"]) * self.dt
        # 宽度限速 -> 每根手指的关节限速。
        self.finger_step = 0.5 * float(ctrl["finger_speed_m_per_s"]) * self.dt
        # 目标位姿（外部设定）与指令位姿（实际送去 IK 的）。
        self.goal_pos = np.zeros(3)
        self.goal_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.cmd_pos = np.zeros(3)
        self.cmd_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.width_target = 0.0
        self.last_ik_error = np.zeros(6)
        self.last_q_command = np.zeros(7)
        self.reset()

    # ------------------------------------------------------------------ 同步
    def reset(self) -> None:
        """把目标与指令位姿同步到当前实测状态（每集复位后第一件事）。"""
        pos, quat = tcp_pose(self.bundle)
        self.goal_pos = pos.copy()
        self.goal_quat = quat_normalize(quat)
        self.cmd_pos = pos.copy()
        self.cmd_quat = self.goal_quat.copy()
        self.width_target = float(self.bundle.gripper_width())
        self.last_ik_error = np.zeros(6)

    def set_pose_target(self, pos, quat) -> None:
        """设定目标位姿；指令位姿由 :meth:`apply` 限速逼近它。"""
        self.goal_pos = np.asarray(pos, dtype=np.float64).reshape(3)
        self.goal_quat = quat_normalize(quat)

    def set_width_target(self, width: float) -> None:
        self.width_target = float(np.clip(width, self.width_min, self.width_max))

    # ------------------------------------------------------------------ 执行
    def _slew_command(self) -> None:
        """指令位姿朝目标推进至多一个限速步。"""
        delta = self.goal_pos - self.cmd_pos
        dist = float(np.linalg.norm(delta))
        if dist > self.max_pos_step:
            delta *= self.max_pos_step / dist
        self.cmd_pos = self.cmd_pos + delta

        err = quat_error_rotvec(self.goal_quat, self.cmd_quat)
        angle = float(np.linalg.norm(err))
        if angle > self.max_rot_step:
            err *= self.max_rot_step / angle
        self.cmd_quat = quat_apply_delta(self.cmd_quat, err)

    def apply(self) -> tuple[np.ndarray, float]:
        """解一个策略步的关节指令（不推进仿真）。

        返回指令化的 ``(eef_target(7,), gripper_target_width)``，供动作记录。
        """
        robot = self.bundle.robot
        self._slew_command()
        q_now = np.asarray(robot.get_dofs_position(dofs_idx_local=self.arm_dofs).detach().cpu())
        q_now = q_now.reshape(-1)
        init_qpos = self.bundle.joint_state()

        q_sol, err = robot.inverse_kinematics(
            link=robot.get_link(self.bundle.hand_link_name),
            pos=self.cmd_pos,
            quat=self.cmd_quat,
            local_point=self.tcp_offset,
            init_qpos=init_qpos,
            respect_joint_limit=True,
            max_samples=self.max_samples,
            max_solver_iters=self.max_iters,
            damping=self.damping,
            max_step_size=self.ik_step,
            seed=0,
            dofs_idx_local=self.arm_dofs,
            return_error=True,
        )
        q_sol = np.asarray(q_sol.detach().cpu(), dtype=np.float64).reshape(-1)
        q_arm = q_sol[self.arm_dofs]
        # IK 之外的保险：单步关节增量再夹一次。
        q_arm = np.clip(q_arm, q_now - MAX_JOINT_STEP_RAD, q_now + MAX_JOINT_STEP_RAD)
        self.last_ik_error = np.asarray(err.detach().cpu(), dtype=np.float64).reshape(-1)[:6]
        self.last_q_command = np.asarray(q_arm, dtype=np.float64).copy()
        robot.control_dofs_position(q_arm, dofs_idx_local=self.arm_dofs)

        # 手指限速逼近目标开口（两个指关节同值指令）。
        q_f = np.asarray(robot.get_dofs_position(dofs_idx_local=self.finger_dofs).detach().cpu())
        q_f = q_f.reshape(-1)
        width_now = float(np.sum(q_f))
        delta = np.clip(self.width_target - width_now, -2.0 * self.finger_step, 2.0 * self.finger_step)
        half = 0.5 * (width_now + delta)
        half = float(np.clip(half, 0.5 * self.width_min, 0.5 * self.width_max))
        finger_cmd = np.full(len(self.finger_dofs), half, dtype=np.float64)
        robot.control_dofs_position(finger_cmd, dofs_idx_local=self.finger_dofs)

        eef = np.concatenate([self.cmd_pos, self.cmd_quat])
        return eef, float(2.0 * half)

    def at_pose(self, pos, quat=None, *, pos_tol: float = 0.01, rot_tol: float = 0.1) -> bool:
        """实测 TCP 是否已到指定位姿（位置 m / 角度 rad；quat 为 None 时只看位置）。"""
        p, q = tcp_pose(self.bundle)
        if np.linalg.norm(p - np.asarray(pos)) > pos_tol:
            return False
        if quat is None:
            return True
        return float(np.linalg.norm(quat_error_rotvec(quat, q))) <= rot_tol

    def settled(self) -> bool:
        """指令位姿是否已追上目标位姿（用于阶段切换的“不急”判断）。"""
        return bool(
            np.linalg.norm(self.goal_pos - self.cmd_pos) <= 1e-4
            and np.linalg.norm(quat_error_rotvec(self.goal_quat, self.cmd_quat)) <= 1e-3
        )


__all__ = [
    "MAX_JOINT_STEP_RAD",
    "DeltaPoseController",
    "ProbePose",
    "point_world",
    "tcp_pose",
]

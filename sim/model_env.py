"""Genesis environment wrapper for Cartesian action chunks.

The policy-facing action is::

    [dx, dy, dz, drot_x, drot_y, drot_z, gripper_width]

The pose part is expressed in the robot base/world frame and is applied to the
current measured TCP pose. ``gripper_width`` is an absolute total opening in
meters. The wrapper converts the resulting absolute TCP target through Genesis
IK and advances the simulator at the configured control rate.

This module deliberately does not import model code. Callers can convert
``observation()`` into their model Observation format, run inference, then pass
one physical 7-D action to ``step()``.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from sim.config import load_dataset_config, load_task_config
from sim.control.controller import DeltaPoseController, tcp_pose
from sim.episode import EpisodeSpec
from sim.obs import get_policy_observation, get_robot_observation
from sim.rotations import quat_apply_delta
from sim.sampler import LiftSampler
from sim.scene import SceneBundle, build_scene

RATE_SMOOTH_S = 0.1


class _ThermalFeatureBuffer:
    """Causal 12-channel thermal window matching model training features."""

    def __init__(self, window_size: int):
        if window_size < 1:
            raise ValueError("window_size must be positive")
        self.window_size = int(window_size)
        self.samples: deque[np.ndarray] = deque(maxlen=self.window_size)
        self.temp_reference: np.ndarray | None = None
        self.smooth_temp: np.ndarray | None = None
        self.last_time: float | None = None

    def reset(self, temp: np.ndarray, *, now_s: float) -> None:
        self.samples.clear()
        self.temp_reference = np.asarray(temp, dtype=np.float32).copy()
        self.smooth_temp = np.asarray(temp, dtype=np.float32).copy()
        self.last_time = float(now_s)
        self.append(temp=np.asarray(temp, dtype=np.float32), force=np.zeros((2, 3)), now_s=now_s)

    def append(self, *, temp: np.ndarray, force: np.ndarray, now_s: float) -> None:
        temp = np.asarray(temp, dtype=np.float32).reshape(2)
        force = np.asarray(force, dtype=np.float32).reshape(2, 3)
        if self.temp_reference is None or self.smooth_temp is None or self.last_time is None:
            raise RuntimeError("ThermalFeatureBuffer.append() called before reset()")

        dt = max(float(now_s) - self.last_time, 1e-6)
        previous_smooth = self.smooth_temp.copy()
        alpha = min(1.0, dt / max(RATE_SMOOTH_S, dt))
        self.smooth_temp += alpha * (temp - self.smooth_temp)
        rate = (self.smooth_temp - previous_smooth) / dt

        sample = np.concatenate(
            [
                temp,
                temp - self.temp_reference,
                rate,
                force.reshape(-1),
            ]
        ).astype(np.float32)
        self.samples.append(sample)
        self.last_time = float(now_s)

    def get(self) -> tuple[np.ndarray, np.ndarray]:
        window = np.zeros((self.window_size, 12), dtype=np.float32)
        mask = np.zeros(self.window_size, dtype=bool)
        count = len(self.samples)
        if count:
            window[-count:] = np.stack(tuple(self.samples))
            mask[-count:] = True
        return window, mask


class GenesisModelEnv:
    """LIBERO-style single-environment wrapper for model-generated actions."""

    action_dim = 7
    state_dim = 8

    def __init__(
        self,
        task_config: str | Path | Mapping[str, Any],
        dataset_config: str | Path | Mapping[str, Any] | None = None,
        *,
        split: str = "train",
        seed: int | None = None,
        max_steps: int = 300,
        thermal_window_size: int = 25,
        thermal_hz: float = 50.0,
        sampler_count: int = 1,
        show_viewer: bool = False,
    ):
        self.task_cfg = (
            load_task_config(task_config)
            if isinstance(task_config, (str, Path))
            else dict(task_config)
        )
        self.dataset_cfg = (
            load_dataset_config(dataset_config)
            if isinstance(dataset_config, (str, Path))
            else dict(dataset_config) if dataset_config is not None else None
        )
        self.split = split
        self.seed = seed
        self.max_steps = int(max_steps)
        self.thermal_hz = float(thermal_hz)
        if self.thermal_hz <= 0:
            raise ValueError("thermal_hz must be positive")

        self.bundle: SceneBundle = build_scene(
            self.task_cfg,
            seed=0 if seed is None else int(seed),
            show_viewer=show_viewer,
        )
        self.controller = DeltaPoseController(self.bundle)
        self.decimation = int(self.task_cfg["controller"]["decimation"])
        self.sim_dt = float(self.task_cfg["controller"]["sim_dt"])
        self.control_dt = self.decimation * self.sim_dt
        self.thermal_interval = 1.0 / self.thermal_hz

        self.sampler = (
            LiftSampler(
                self.task_cfg,
                self.dataset_cfg,
                split,
                seed=seed,
                count=int(sampler_count),
            )
            if self.dataset_cfg is not None
            else None
        )
        self.thermal = _ThermalFeatureBuffer(int(thermal_window_size))

        self.spec: EpisodeSpec | None = None
        self.sim_step = 0
        self.policy_step = 0
        self.sim_time = 0.0
        self.next_thermal_time = 0.0
        self.last_action: np.ndarray | None = None
        self.last_command: tuple[np.ndarray, float] | None = None

    def reset(
        self,
        index: int = 0,
        *,
        spec: EpisodeSpec | None = None,
    ) -> dict[str, np.ndarray | str]:
        """Reset the Genesis scene and initialize the thermal window."""
        if spec is None:
            if self.sampler is None:
                raise ValueError("reset() needs spec when dataset_config is not provided")
            spec = self.sampler.sample(int(index))

        self.spec = spec
        self.bundle.reset_episode(spec)
        self.controller.reset()

        self.sim_step = 0
        self.policy_step = 0
        self.sim_time = 0.0
        self.next_thermal_time = self.thermal_interval
        self.last_action = None
        self.last_command = None

        sensor = get_policy_observation(self.bundle)
        self.thermal.reset(sensor["fingertip_temp"], now_s=self.sim_time)
        return self.observation()

    @property
    def done(self) -> bool:
        return self.policy_step >= self.max_steps

    def observation(self) -> dict[str, np.ndarray | str]:
        """Return the policy-facing observation without model-specific tensors."""
        robot = get_robot_observation(self.bundle)
        camera = self.bundle.camera_observation()

        joint_position = np.asarray(robot["joint_position"], dtype=np.float32)[:7]
        gripper_width = np.float32(robot["gripper_width"])
        state = np.concatenate([joint_position, [gripper_width]]).astype(np.float32)

        thermal_window, thermal_mask = self.thermal.get()
        base_rgb = np.asarray(camera["base_0_rgb"], dtype=np.uint8)
        wrist_rgb = np.asarray(camera["wrist_0_rgb"], dtype=np.uint8)
        right_wrist_rgb = np.zeros_like(base_rgb)

        return {
            "image": {
                "base_0_rgb": base_rgb,
                "left_wrist_0_rgb": wrist_rgb,
                "right_wrist_0_rgb": right_wrist_rgb,
            },
            "image_mask": {
                "base_0_rgb": np.asarray(True),
                "left_wrist_0_rgb": np.asarray(True),
                "right_wrist_0_rgb": np.asarray(False),
            },
            "state": state,
            "prompt": str(self.spec.instruction if self.spec is not None else ""),
            "thermal_window": thermal_window,
            "thermal_window_time_mask": thermal_mask,
            # Useful for diagnostics; model adapters may ignore these.
            "eef_pose": np.asarray(robot["eef_pose"], dtype=np.float32),
            "joint_position": np.asarray(robot["joint_position"], dtype=np.float32),
            "gripper_width": np.asarray(gripper_width, dtype=np.float32),
        }

    def state_to_fingertip_geometry(
        self,
        state: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return [B,2,3] pad positions and normals for the current FK state.

        Genesis exposes link poses resulting from the same joint state recorded
        in ``observation["state"]``. A real-robot adapter should replace this
        method with URDF/Pinocchio FK while keeping the returned contract.
        """
        state = np.asarray(state, dtype=np.float32)
        if state.ndim == 1:
            state = state[None, :]
        if state.ndim != 2 or state.shape[-1] < 8:
            raise ValueError(f"state must be [B,>=8], got {state.shape}")
        policy = get_policy_observation(self.bundle)
        positions = np.asarray(policy["sensor_position_base"], dtype=np.float32)
        normals = np.asarray(policy["sensor_normal_base"], dtype=np.float32)
        batch = state.shape[0]
        return (
            np.repeat(positions[None], batch, axis=0),
            np.repeat(normals[None], batch, axis=0),
        )

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray | str], bool, dict[str, Any]]:
        """Execute one physical 7-D action and advance one policy control step."""
        if self.spec is None:
            raise RuntimeError("reset() must be called before step()")
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape != (self.action_dim,):
            raise ValueError(f"action must have shape (7,), got {action.shape}")
        if not np.all(np.isfinite(action)):
            raise ValueError("action contains NaN or Inf")

        pos_now, quat_now = tcp_pose(self.bundle)
        target_pos = pos_now + action[:3]
        target_quat = quat_apply_delta(quat_now, action[3:6])

        self.controller.set_pose_target(target_pos, target_quat)
        self.controller.set_width_target(float(action[6]))
        self.last_command = self.controller.apply()
        self.last_action = action.copy()

        for _ in range(self.decimation):
            self.bundle.step()
            self.sim_step += 1
            self.sim_time += self.sim_dt
            while self.sim_time + 1e-12 >= self.next_thermal_time:
                sensor = get_policy_observation(self.bundle)
                self.thermal.append(
                    temp=sensor["fingertip_temp"],
                    force=sensor["contact_force_link"],
                    now_s=self.sim_time,
                )
                self.next_thermal_time += self.thermal_interval

        self.policy_step += 1
        info = {
            "policy_step": self.policy_step,
            "sim_step": self.sim_step,
            "sim_time_s": self.sim_time,
            "command_eef": (
                None if self.last_command is None else self.last_command[0].copy()
            ),
            "command_gripper": (
                None if self.last_command is None else self.last_command[1]
            ),
        }
        return self.observation(), self.done, info


__all__ = ["GenesisModelEnv"]

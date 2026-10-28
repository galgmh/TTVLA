"""The policy I/O contract shared by the exporter and the closed-loop eval.

Deliberately free of ``sim`` / Genesis imports: the LeRobot exporter runs on the
data machine, where the simulator stack is not needed (only h5py + numpy).

State layout (``observation.state`` in the LeRobot dataset, and the policy's
``PolicyObservation.state`` at evaluation time)::

    proprio16   [0:3]    ee_pos            world TCP position, m
                [3:6]    ee_ori            world TCP rotation vector, rad
                [6:7]    gripper_states     -1 open .. +1 closed
                [7:16]   joint_states       7 arm joints + 2 finger joints

    touch28     proprio16
                [16:28]  tactile/force      2 fingers x [Fn(3), Ft(3)], finger-major

    full30      proprio16
                [16:18]  temperature/fingertip_grid   measured cell per finger, C
                [18:30]  tactile/force

Actions are the 7-dim LIBERO-style delta the controller consumes::

    a[0:3] * controller.pos_scale_m   -> world TCP translation delta
    a[3:6] * controller.rot_scale_rad -> world TCP rotation-vector delta
    a[6]                              -> gripper synergy, -1 open .. +1 closed

Only *policy-visible* channels are assembled here. The scene's ground-truth
temperature (``temperature/object``, ``temperature/*_grid``, ``link_temps``) and
the object poses with their fixed row order are intentionally absent: they leak
the answer, and a policy that sees them is not the baseline we are measuring.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# State modes and their exact widths. The names double as the LeRobot feature
# comment (`info.json` -> features["observation.state"]["names"]).
STATE_DIMS: dict[str, int] = {"proprio16": 16, "touch28": 28, "full30": 30}

_PROPRIO_NAMES = [
    "ee_pos_x",
    "ee_pos_y",
    "ee_pos_z",
    "ee_ori_x",
    "ee_ori_y",
    "ee_ori_z",
    "gripper",
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "joint_7",
    "finger_1",
    "finger_2",
]
_FINGERTIP_NAMES = ["finger0_temp_c", "finger1_temp_c"]
# `tactile/force` is (n_fingers, 6) = [Fn(3), Ft(3)] per finger, finger-major.
_FORCE_NAMES = [
    f"finger{f}_force_{axis}"
    for f in (0, 1)
    for axis in ("n_x", "n_y", "n_z", "t_x", "t_y", "t_z")
]

STATE_NAMES: dict[str, list[str]] = {
    "proprio16": list(_PROPRIO_NAMES),
    "touch28": _PROPRIO_NAMES + _FORCE_NAMES,
    "full30": _PROPRIO_NAMES + _FINGERTIP_NAMES + _FORCE_NAMES,
}

ACTION_DIM = 7
ACTION_NAMES = [
    "dpos_x",
    "dpos_y",
    "dpos_z",
    "drot_x",
    "drot_y",
    "drot_z",
    "gripper",
]

# LeRobot feature keys written by the exporter. They match openpi's LIBERO data
# config, so a π0.5 fine-tune can start from `LeRobotLiberoDataConfig` with only
# the repo path / state width changed.
IMAGE_KEYS: tuple[str, ...] = (
    "observation.images.image",  # agentview (fixed third-person view)
    "observation.images.wrist_image",  # eye-in-hand
)


def check_state_mode(mode: str) -> str:
    if mode not in STATE_DIMS:
        raise ValueError(
            f"unknown state mode {mode!r}; choose one of {sorted(STATE_DIMS)}"
        )
    return mode


def state_dim(mode: str) -> int:
    return STATE_DIMS[check_state_mode(mode)]


def state_names(mode: str) -> list[str]:
    return list(STATE_NAMES[check_state_mode(mode)])


def _rows(value: Any, what: str) -> np.ndarray:
    """Time-major ``(T, k)`` view of one channel (a 1-D value is one step)."""
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        raise ValueError(f"{what} must have a time/probe axis, got a scalar")
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr.reshape(arr.shape[0], -1)


def assemble_state_arrays(
    ee_pos: Any,
    ee_ori: Any,
    gripper_states: Any,
    joint_states: Any,
    *,
    fingertip_grid: Any | None = None,
    tactile_force: Any | None = None,
    mode: str = "proprio16",
) -> np.ndarray:
    """Stack the channels in the documented order; returns ``(T, D)`` float32.

    Every argument is time-major (``(T, k)``, or ``(T, *grid)`` for the
    fingertip grid). The exporter passes whole episodes; the eval passes one
    step at a time.
    """
    mode = check_state_mode(mode)
    if mode in ("touch28", "full30") and tactile_force is None:
        raise ValueError(f"state mode {mode!r} needs tactile/force")
    if mode == "full30" and fingertip_grid is None:
        raise ValueError("state mode 'full30' needs temperature/fingertip_grid")

    parts = [
        _rows(ee_pos, "ee_pos"),
        _rows(ee_ori, "ee_ori"),
        _rows(gripper_states, "gripper_states"),
        _rows(joint_states, "joint_states"),
    ]
    if mode == "full30":
        parts.append(_rows(fingertip_grid, "fingertip_grid"))
    if mode in ("touch28", "full30"):
        parts.append(_rows(tactile_force, "tactile_force"))

    out = np.concatenate(parts, axis=-1).astype(np.float32)
    want = state_dim(mode)
    if out.shape[-1] != want:
        raise ValueError(
            f"assembled {out.shape[-1]} dims for {mode!r}, expected {want}"
        )
    return out


def assemble_state_from_obs(obs: dict[str, Any], mode: str = "proprio16") -> np.ndarray:
    """One policy-rate observation dict (``sim.obs.ObsBuilder.build``) -> ``(D,)``.

    Only the channels in the selected mode are read; the thermal channels are
    the *measured* fingertip cell (``temperature/fingertip_grid``), never the
    object grid or the link-temperature table.
    """
    mode = check_state_mode(mode)
    tactile = obs.get("tactile", {}) or {}
    temperature = obs.get("temperature", {}) or {}

    fingertip = None
    if mode == "full30":
        if "fingertip_grid" not in temperature:
            raise KeyError(
                "full30 needs obs['temperature']['fingertip_grid']; "
                "was the temperature sensor built?"
            )
        # (n_fingers, 1, 1, 1) -> (1, n_fingers)
        fingertip = np.asarray(temperature["fingertip_grid"], dtype=np.float32).reshape(1, -1)

    force = None
    if mode in ("touch28", "full30"):
        if "force" not in tactile:
            raise KeyError(
                f"{mode} needs obs['tactile']['force']; add 'force' to "
                "`tactile.modalities` in the task YAML"
            )
        force = np.asarray(tactile["force"], dtype=np.float32).reshape(1, -1)

    state = assemble_state_arrays(
        obs["ee_pos"],
        obs["ee_ori"],
        obs["gripper_states"],
        obs["joint_states"],
        fingertip_grid=fingertip,
        tactile_force=force,
        mode=mode,
    )
    return state[0]


__all__ = [
    "ACTION_DIM",
    "ACTION_NAMES",
    "IMAGE_KEYS",
    "STATE_DIMS",
    "STATE_NAMES",
    "assemble_state_arrays",
    "assemble_state_from_obs",
    "check_state_mode",
    "state_dim",
    "state_names",
]

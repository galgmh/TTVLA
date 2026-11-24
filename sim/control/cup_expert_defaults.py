"""Cup expert geometry and low-level execution defaults.

These values belong to the scripted expert pipeline.  They are deliberately
kept out of task YAML files: a task declares the goal and success criterion,
while an expert is only one way of producing a demonstration for that task.
All positions are expressed in the cup asset authoring frame.
"""

from __future__ import annotations

EXPERT_REGIONS = {
    "lower": {
        "pos": [0.0, 0.0, 0.020], "tool_dir": [-1.0, 0.0, 0.0],
        "finger_dir": [0.0, -1.0, 0.0], "pre_dist": 0.06,
        "xy_jitter_m": 0.002, "z_jitter_m": 0.0015,
        "azimuth_jitter_rad": 0.35, "tilt_jitter_rad": 0.12,
    },
    "middle": {
        "pos": [0.0, 0.0, 0.055], "tool_dir": [-1.0, 0.0, 0.0],
        "finger_dir": [0.0, -1.0, 0.0], "pre_dist": 0.06,
        "xy_jitter_m": 0.002, "z_jitter_m": 0.0015,
        "azimuth_jitter_rad": 0.35, "tilt_jitter_rad": 0.12,
    },
    "upper": {
        "pos": [0.0, 0.0, 0.088], "tool_dir": [0.0, 0.0, -1.0],
        "finger_dir": [0.0, -1.0, 0.0], "pre_dist": 0.08,
        "xy_jitter_m": 0.002, "z_jitter_m": 0.001,
        "azimuth_jitter_rad": 0.35, "tilt_jitter_rad": 0.10,
    },
    "handle": {
        "pos": [-0.057, 0.0, 0.056], "tool_dir": [1.0, 0.0, 0.0],
        "finger_dir": [0.0, -1.0, 0.0], "pre_dist": 0.07,
        "xy_jitter_m": 0.0008, "z_jitter_m": 0.0008,
        "azimuth_jitter_rad": 0.0, "tilt_jitter_rad": 0.08,
    },
}

MIXED_PROBE = {
    "pos": [0.0, 0.0, 0.0733], "tool_dir": [-1.0, 0.0, 0.0],
    "finger_dir": [0.0, -1.0, 0.0], "pre_dist": 0.06,
    "xy_jitter_m": 0.0005, "z_jitter_m": 0.0003,
}

EXPERT_TIMING = {
    "contact_s": 0.50,
    "post_contact_s": 0.20,
    "retract_open_timeout_s": 3.0,
    "retract_move_timeout_s": 5.0,
}

EXPERT_CONTACT = {
    "seek_timeout_s": 8.0,
    "brief_hold_s": [0.08, 0.18],
    "sample_every_sim_steps": 2,
}

__all__ = [
    "EXPERT_CONTACT", "EXPERT_REGIONS", "EXPERT_TIMING", "MIXED_PROBE",
]

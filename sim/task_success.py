"""Task-level success evaluation independent from expert termination states."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def evaluate_task_success(
    task_cfg: Mapping[str, Any],
    buffers: Mapping[str, Sequence[Any]],
    *,
    initial_cup_z: float,
    final_cup_z: float,
    region_order: Sequence[str],
    initial_link_temps_c: Sequence[float],
) -> dict[str, Any]:
    """Evaluate the two required conditions: grasp region and lift height."""
    criterion = dict(task_cfg["success"])
    lift_delta = float(final_cup_z) - float(initial_cup_z)
    lift_ok = lift_delta >= float(criterion["min_lift_height_m"])

    masks = np.asarray(buffers["privileged/contact_link_mask"], dtype=bool)
    bilateral = masks.any(axis=2).all(axis=1) if masks.size else np.zeros(0, dtype=bool)
    final_mask = masks[np.flatnonzero(bilateral)[-1]] if np.any(bilateral) else None
    final_sensor_index = int(np.flatnonzero(bilateral)[-1]) if np.any(bilateral) else None
    contacted: list[str] = []
    if final_mask is not None:
        contacted = [
            str(region_order[i])
            for i in np.flatnonzero(final_mask.any(axis=0))
        ]

    required = str(criterion["required_grasp_region"])
    if required == "thermally_safe":
        candidates = set(str(v) for v in criterion["candidate_contact_regions"])
        max_temp = float(criterion["max_grasp_temperature_c"])
        temps = {str(name): float(temp) for name, temp in zip(region_order, initial_link_temps_c)}
        region_ok = bool(contacted) and all(
            name in candidates and temps[name] <= max_temp for name in contacted
        )
    else:
        allowed = set(str(v) for v in criterion["allowed_contact_regions"])
        region_ok = bool(contacted) and all(name in allowed for name in contacted)
        if region_ok and "contact_height_range_m" in criterion:
            positions = np.asarray(buffers["tactile/sensor_position_base"], dtype=float)
            lo, hi = (float(v) for v in criterion["contact_height_range_m"])
            relative_z = positions[final_sensor_index, :, 2] - float(final_cup_z)
            region_ok = bool(np.all((relative_z >= lo) & (relative_z <= hi)))

    return {
        "success": bool(region_ok and lift_ok),
        "grasp_region_success": bool(region_ok),
        "lift_success": bool(lift_ok),
        "grasp_contact_regions": contacted,
        "lift_height_m": lift_delta,
        "required_grasp_region": required,
    }


__all__ = ["evaluate_task_success"]

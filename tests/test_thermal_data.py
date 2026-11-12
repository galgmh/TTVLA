"""在线热触觉数据接口测试。"""

import numpy as np
import torch

from training.pi05.thermal.labels import build_thermal_supervision
from training.pi05.thermal.transforms import (
    THERMAL_OBSERVATION_KEYS, EmptyThermalObservation, ThermalObservationTransform,
    thermal_supervision_from_batch,
)


def _supervision(batch=2, queries=4):
    return {
        "contact_temp_c": np.zeros((batch, 1), np.float32),
        "contact_valid": np.ones((batch, 1), np.float32),
        "field_query_xyz_base": np.zeros((batch, queries, 3), np.float32),
        "field_query_temp_c": np.zeros((batch, queries, 1), np.float32),
        "field_query_mask": np.ones((batch, queries), bool),
        "strategy_type": np.arange(batch, dtype=np.int64) % 4,
    }


def test_observation_contains_only_current_window():
    assert THERMAL_OBSERVATION_KEYS == (
        "thermal_window", "thermal_window_time_mask", "sensor_xyz_base",
        "sensor_normal_base",
    )
    empty = EmptyThermalObservation(window_size=4, signal_dim=6)({})
    assert empty["thermal_window"].shape == (4, 6)
    assert empty["thermal_window_time_mask"].shape == (4,)
    assert not any("history" in key or "contact_mask" in key for key in empty)


def test_transform_dtypes():
    data = {key: np.ones(2) for key in THERMAL_OBSERVATION_KEYS}
    data["thermal_window_time_mask"] = np.array([0, 1])
    output = ThermalObservationTransform()(data)
    assert output["thermal_window"].dtype == np.float32
    assert output["thermal_window_time_mask"].dtype == np.bool_


def test_supervision_is_current_contact_only():
    supervision = build_thermal_supervision(_supervision())
    assert supervision.contact_temp_c.shape == (2, 1)
    assert supervision.contact_valid.shape == (2, 1)
    assert supervision.field_query_mask.dtype == torch.bool
    assert thermal_supervision_from_batch({"thermal_supervision": _supervision()}) is not None

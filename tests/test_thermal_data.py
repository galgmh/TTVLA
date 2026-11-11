"""数据侧工具测试：历史缓存、候选点、标签、transform 与闭环驱动（Task G）。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from training.pi05.thermal.history import ContactHistoryBuffer
from training.pi05.thermal.history import merge_snapshot
from training.pi05.thermal.labels import SKILL_GRASP
from training.pi05.thermal.labels import SKILL_OTHER
from training.pi05.thermal.labels import SKILL_PROBE
from training.pi05.thermal.labels import assign_region_temperatures
from training.pi05.thermal.labels import build_thermal_supervision
from training.pi05.thermal.labels import safe_from_temperature
from training.pi05.thermal.rollout import CompletedContact
from training.pi05.thermal.rollout import ThermalClosedLoop
from training.pi05.thermal.surfaces import build_surface_candidates
from training.pi05.thermal.transforms import THERMAL_OBSERVATION_KEYS
from training.pi05.thermal.transforms import ThermalObservationTransform
from training.pi05.thermal.transforms import thermal_supervision_from_batch


def _signal(value: float, length: int = 4, channels: int = 2) -> np.ndarray:
    return np.full((length, channels), value, dtype=np.float32)


# ---------------------------------------------------------------- 历史缓存


def test_history_empty_snapshot_shapes_and_masks():
    history = ContactHistoryBuffer(history_size=3, window_size=4, signal_dim=2)
    snapshot = history.snapshot(2.0)

    assert snapshot["thermal_history"].shape == (3, 4, 2)
    assert snapshot["thermal_history_time_mask"].shape == (3, 4)
    assert snapshot["contact_mask"].shape == (3,)
    assert snapshot["contact_xyz_obj"].shape == (3, 3)
    assert snapshot["contact_normal_obj"].shape == (3, 3)
    assert snapshot["contact_age_s"].shape == (3, 1)
    assert not snapshot["contact_mask"].any()
    assert not snapshot["thermal_history_time_mask"].any()
    assert np.all(snapshot["thermal_history"] == 0)


def test_history_add_right_aligns_and_computes_age():
    history = ContactHistoryBuffer(history_size=3, window_size=4, signal_dim=2)
    history.add_completed_contact(_signal(1.0), np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 2.0]), 1.0)

    snapshot = history.snapshot(2.5)
    assert snapshot["contact_mask"].tolist() == [False, False, True]
    assert np.all(snapshot["thermal_history"][2] == 1.0)
    assert snapshot["thermal_history_time_mask"][2].all()
    assert snapshot["contact_age_s"][2, 0] == pytest.approx(1.5)
    # 法向写入时归一化。
    np.testing.assert_allclose(snapshot["contact_normal_obj"][2], [0.0, 0.0, 1.0])

    history.add_completed_contact(_signal(2.0), np.array([4.0, 5.0, 6.0]), np.array([1.0, 0.0, 0.0]), 2.0)
    snapshot = history.snapshot(2.0)
    assert snapshot["contact_mask"].tolist() == [False, True, True]
    assert snapshot["contact_age_s"][1, 0] == pytest.approx(1.0)
    assert snapshot["contact_age_s"][2, 0] == pytest.approx(0.0)


def test_history_overflow_drops_oldest():
    history = ContactHistoryBuffer(history_size=2, window_size=4, signal_dim=2)
    for index, value in enumerate((1.0, 2.0, 3.0)):
        history.add_completed_contact(_signal(value), np.zeros(3), np.array([0.0, 0.0, 1.0]), float(index))

    snapshot = history.snapshot(3.0)
    assert snapshot["contact_mask"].tolist() == [True, True]
    assert np.all(snapshot["thermal_history"][0] == 2.0)
    assert np.all(snapshot["thermal_history"][1] == 3.0)
    assert snapshot["contact_age_s"][0, 0] == pytest.approx(2.0)


def test_history_rejects_invalid_records():
    history = ContactHistoryBuffer(history_size=2, window_size=4, signal_dim=2)
    with pytest.raises(ValueError, match="形状"):
        history.add_completed_contact(_signal(0.0, length=3), np.zeros(3), np.array([0.0, 0.0, 1.0]), 0.0)
    with pytest.raises(ValueError, match="normal"):
        history.add_completed_contact(_signal(0.0), np.zeros(3), np.zeros(3), 0.0)
    bad = _signal(0.0)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="有限"):
        history.add_completed_contact(bad, np.zeros(3), np.array([0.0, 0.0, 1.0]), 0.0)
    with pytest.raises(ValueError, match="history_size"):
        ContactHistoryBuffer(history_size=0)


def test_merge_snapshot_does_not_mutate_base():
    history = ContactHistoryBuffer(history_size=1, window_size=2, signal_dim=1)
    base = {"image": 1}
    merged = merge_snapshot(base, history.snapshot(0.0))
    assert set(base) == {"image"}
    assert "contact_mask" in merged


# ---------------------------------------------------------------- 候选表面点


def test_surface_candidates_sampling_and_normalization():
    rng = np.random.default_rng(0)
    points = rng.uniform(-0.05, 0.05, size=(20, 3)).astype(np.float32)
    normals = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (20, 1))
    candidates = build_surface_candidates(
        points,
        normals,
        np.zeros(3),
        num_points=8,
        rng=np.random.default_rng(1),
        reach_radius_m=0.03,
    )

    assert candidates.xyz_obj.shape == (8, 3)
    assert candidates.mask.all()
    assert candidates.reachable.dtype == np.bool_
    assert candidates.grasp_score.shape == (8, 1)
    assert np.max(np.abs(candidates.xyz_obj)) <= 1.0 + 1e-5
    np.testing.assert_allclose(candidates.normal_obj, np.tile([0.0, 0.0, 1.0], (8, 1)))
    np.testing.assert_allclose(candidates.grasp_score, 1.0)
    # 启发式可达：距离 <= 0.03（全部点在原点附近 0.05 内，部分不可达）。
    assert candidates.reachable.any()
    assert not candidates.reachable.all()


def test_surface_candidates_padding_when_fewer_points():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    normals = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), (3, 1))
    candidates = build_surface_candidates(points, normals, np.zeros(3), num_points=6)

    assert candidates.mask.tolist() == [True, True, True, False, False, False]
    assert not candidates.reachable[3:].any()
    assert np.all(candidates.grasp_score[3:] == 0.0)
    np.testing.assert_allclose(candidates.normal_obj[3:], 0.0)


def test_surface_candidates_exact_labels_take_precedence():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    normals = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (4, 1))
    reachable = np.array([0, 0, 1, 0], dtype=bool)
    scores = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    candidates = build_surface_candidates(
        points, normals, np.zeros(3), num_points=4, reachable=reachable, grasp_score=scores
    )

    assert candidates.reachable.tolist() == [False, False, True, False]
    # 精确分数在全部有效点上保留（可达与否不改变分数）。
    np.testing.assert_allclose(candidates.grasp_score.reshape(-1), scores)


# ---------------------------------------------------------------- 监督标签


def test_assign_region_temperatures_by_nearest_center():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
        ],
        dtype=np.float32,
    )
    temps, mask = assign_region_temperatures(
        points,
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        np.array([10.0, 30.0], dtype=np.float32),
        radius_m=0.3,
    )

    assert mask.tolist() == [True, True, True, False]
    assert temps.tolist()[0] == pytest.approx(10.0)
    assert temps.tolist()[1] == pytest.approx(10.0)
    assert temps.tolist()[2] == pytest.approx(30.0)
    assert temps.tolist()[3] == pytest.approx(0.0)


def test_safe_from_temperature_and_mask():
    temps = np.array([5.0, 20.0, 40.0, 5.0], dtype=np.float32)
    mask = np.array([True, True, True, False])
    safe = safe_from_temperature(temps, mask, safe_max_c=15.0)

    assert safe.shape == (4, 1)
    assert safe.reshape(-1).tolist() == [1.0, 0.0, 0.0, 0.0]


def test_build_thermal_supervision_roundtrip_and_validation():
    arrays = {
        "contact_temp_c": np.zeros((2, 3, 1), dtype=np.float32),
        "contact_valid": np.ones((2, 3, 1), dtype=np.float32),
        "surface_temp_c": np.zeros((2, 4, 1), dtype=np.float32),
        "surface_temp_mask": np.ones((2, 4), dtype=bool),
        "surface_safe": np.ones((2, 4, 1), dtype=np.float32),
        "strategy_type": np.array([SKILL_PROBE, SKILL_GRASP]),
        "target_surface_index": np.array([0, 3]),
    }
    supervision = build_thermal_supervision(arrays)
    assert supervision.strategy_type.dtype == torch.long
    assert supervision.surface_temp_mask.dtype == torch.bool

    bad = dict(arrays, strategy_type=np.array([SKILL_OTHER, 3]))
    with pytest.raises(ValueError, match="strategy_type"):
        build_thermal_supervision(bad)

    bad = dict(arrays, target_surface_index=np.array([0, 4]))
    with pytest.raises(ValueError, match="target_surface_index"):
        build_thermal_supervision(bad)

    missing = dict(arrays)
    missing.pop("contact_valid")
    with pytest.raises(ValueError, match="缺少监督字段"):
        build_thermal_supervision(missing)


# ---------------------------------------------------------------- transform


def test_thermal_observation_transform_maps_and_casts():
    key_map = {f"src/{key}": key for key in THERMAL_OBSERVATION_KEYS}
    data = {}
    for key in THERMAL_OBSERVATION_KEYS:
        if key in {"thermal_history_time_mask", "contact_mask", "surface_mask", "surface_reachable"}:
            data[f"src/{key}"] = np.array([0, 1], dtype=np.int64)
        else:
            data[f"src/{key}"] = np.ones(2, dtype=np.float64)
    data["image"] = "untouched"

    output = ThermalObservationTransform(key_map=key_map)(data)
    for key in THERMAL_OBSERVATION_KEYS:
        assert key in output
    assert output["contact_mask"].dtype == np.bool_
    assert output["thermal_history"].dtype == np.float32
    assert output["image"] == "untouched"
    assert "src/contact_mask" in output  # 原始键保留

    with pytest.raises(ValueError, match="缺少热觉字段"):
        ThermalObservationTransform(key_map=key_map)({"src/contact_mask": [1]})
    tolerant = ThermalObservationTransform(key_map=key_map, require_all=False)
    assert tolerant({"src/contact_mask": [1]})["contact_mask"].dtype == np.bool_


def test_thermal_supervision_from_batch():
    assert thermal_supervision_from_batch({"actions": np.zeros(1)}) is None
    batch = {
        "thermal_supervision": {
            "contact_temp_c": np.zeros((1, 2, 1)),
            "contact_valid": np.ones((1, 2, 1)),
            "surface_temp_c": np.zeros((1, 3, 1)),
            "surface_temp_mask": np.ones((1, 3), dtype=bool),
            "surface_safe": np.ones((1, 3, 1)),
            "strategy_type": np.array([1]),
            "target_surface_index": np.array([2]),
        }
    }
    supervision = thermal_supervision_from_batch(batch)
    assert supervision is not None
    assert supervision.target_surface_index.tolist() == [2]


# ---------------------------------------------------------------- 闭环驱动


class _RecordingPolicy:
    """按脚本返回技能与选点，并记录每个控制时刻看到的接触数。"""

    def __init__(self, skills: list[int], indices: list[int]):
        self.skills = skills
        self.indices = indices
        self.visible_contacts: list[int] = []

    def sample_actions(self, device, observation, num_steps: int = 10, *, return_diagnostics: bool = True):
        self.visible_contacts.append(int(observation["snapshot"]["contact_mask"].sum()))
        step = len(self.visible_contacts) - 1
        probabilities = torch.zeros(1, 3)
        probabilities[0, self.skills[step]] = 1.0
        diagnostics = {
            "skill_probability": probabilities,
            "selected_surface_index": torch.tensor([self.indices[step]]),
        }
        return torch.zeros(1, 4, 7), diagnostics


class _FakeEnvironment:
    """执行 probe 后返回一个完成的接触窗口；grasp 结束后置 done。"""

    def __init__(self):
        self._now = 0.0
        self._grasped = False
        self.executed = 0

    @property
    def now_s(self) -> float:
        return self._now

    def build_observation(self, snapshot):
        return {"snapshot": dict(snapshot)}

    def execute(self, actions, diagnostics):
        self.executed += 1
        skill = int(diagnostics["skill_probability"][0].argmax())
        self._now += 0.5
        if skill == SKILL_PROBE:
            return CompletedContact(
                signal=_signal(float(self.executed)),
                xyz_obj=np.array([0.0, 0.0, float(self.executed)], dtype=np.float32),
                normal_obj=np.array([0.0, 0.0, 1.0], dtype=np.float32),
                end_time_s=self._now,
            )
        if skill == SKILL_GRASP:
            self._grasped = True
        return None

    def done(self) -> bool:
        return self._grasped


def test_closed_loop_records_contacts_causally():
    policy = _RecordingPolicy(skills=[SKILL_PROBE, SKILL_PROBE, SKILL_GRASP], indices=[1, 2, 0])
    environment = _FakeEnvironment()
    history = ContactHistoryBuffer(history_size=3, window_size=4, signal_dim=2)

    result = ThermalClosedLoop(
        policy=policy,
        environment=environment,
        history=history,
        num_denoise_steps=5,
    ).run()

    # 因果性：第 t 步只看到第 t 步之前完成的接触。
    assert policy.visible_contacts == [0, 1, 2]
    assert [step.selected_surface_index for step in result.steps] == [1, 2, 0]
    assert result.probe_steps == 2
    assert result.grasp_steps == 1
    assert result.completed_contacts == 2
    assert len(history) == 2

    # 最终记忆：两次 probe 都保留（右对齐），且年龄随结束时间递减。
    snapshot = history.snapshot(1.0)
    assert snapshot["contact_mask"].tolist() == [False, True, True]
    assert snapshot["contact_age_s"][1, 0] == pytest.approx(0.5)
    assert snapshot["contact_age_s"][2, 0] == pytest.approx(0.0)


def test_closed_loop_respects_max_steps():
    policy = _RecordingPolicy(skills=[SKILL_PROBE] * 4, indices=[0] * 4)
    environment = _FakeEnvironment()
    history = ContactHistoryBuffer(history_size=2, window_size=4, signal_dim=2)

    result = ThermalClosedLoop(policy=policy, environment=environment, history=history).run(max_steps=2)
    assert len(result.steps) == 2
    assert environment.executed == 2

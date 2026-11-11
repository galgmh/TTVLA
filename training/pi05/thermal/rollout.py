"""Thermal-π0.5 闭环推理驱动（规格第 7.5 / 8 节，Task G）。

一次控制时刻的完整流程：

1. 从 ``ContactHistoryBuffer`` 取 t 时刻可见的历史（只含**已完成**接触）；
2. 环境用该历史拼出 ``Observation``（图像、语言、状态、候选点等）；
3. ``sample_actions(..., return_diagnostics=True)`` 输出 7D action chunk 与
   技能/选点诊断；
4. 执行 action chunk；probe 结束时环境返回已完成接触，驱动把它写入 memory；
5. 进入下一个控制时刻。

因果性由调用顺序保证：Observation 在 ``execute`` 之前构建，本次 probe 的结果
只可能出现在下一次 snapshot 中。本模块不依赖 Genesis——环境只需实现
``now_s`` / ``build_observation`` / ``execute`` / ``done`` 四个接口。
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
from typing import Any, Protocol

import numpy as np

from training.pi05.thermal.history import ContactHistoryBuffer
from training.pi05.thermal.labels import SKILL_GRASP
from training.pi05.thermal.labels import SKILL_PROBE

#: 闭环允许的最长控制步数（防止策略不收敛时死循环）。
DEFAULT_MAX_STEPS = 64


@dataclasses.dataclass
class CompletedContact:
    """一次已完成的接触（probe 或抓取），由环境在动作执行结束后返回。"""

    signal: np.ndarray  # [L, C]
    xyz_obj: np.ndarray  # [3]
    normal_obj: np.ndarray  # [3]
    end_time_s: float


class PolicyProtocol(Protocol):
    """与 ``PI0Pytorch.sample_actions`` 兼容的最小接口。"""

    def sample_actions(
        self,
        device,
        observation,
        num_steps: int = 10,
        *,
        return_diagnostics: bool = True,
    ) -> tuple[Any, Mapping[str, Any]]: ...


class EnvironmentProtocol(Protocol):
    """仿真器/真实环境需要实现的最小闭环接口。"""

    @property
    def now_s(self) -> float: ...

    def build_observation(self, snapshot: Mapping[str, np.ndarray]): ...

    def execute(self, actions, diagnostics: Mapping[str, Any]) -> CompletedContact | None: ...

    def done(self) -> bool: ...


@dataclasses.dataclass(frozen=True)
class RolloutStep:
    """一个控制时刻的日志。"""

    now_s: float
    skill: int
    selected_surface_index: int
    visible_contacts: int
    recorded_contact: bool


@dataclasses.dataclass
class RolloutResult:
    """一次短 episode 的汇总；用于验证因果记忆与选点行为。"""

    steps: list[RolloutStep]

    @property
    def completed_contacts(self) -> int:
        return sum(step.recorded_contact for step in self.steps)

    @property
    def probe_steps(self) -> int:
        return sum(step.skill == SKILL_PROBE for step in self.steps)

    @property
    def grasp_steps(self) -> int:
        return sum(step.skill == SKILL_GRASP for step in self.steps)


def _diagnostic_skill_and_index(diagnostics: Mapping[str, Any]) -> tuple[int, int]:
    """从诊断里取技能编号与选中候选点；缺省时返回 -1。"""
    skill_probability = diagnostics.get("skill_probability")
    if skill_probability is None:
        return -1, -1
    skill_tensor = skill_probability[0] if hasattr(skill_probability, "__getitem__") else skill_probability
    skill = int(skill_tensor.detach().argmax()) if hasattr(skill_tensor, "detach") else int(np.argmax(skill_tensor))
    selected = diagnostics.get("selected_surface_index")
    if selected is None:
        return skill, -1
    first = selected[0] if hasattr(selected, "__getitem__") else selected
    index = int(first.detach()) if hasattr(first, "detach") else int(first)
    return skill, index


class ThermalClosedLoop:
    """驱动策略与环境交互，并把完成的接触因果地写入记忆。"""

    def __init__(
        self,
        *,
        policy: PolicyProtocol,
        environment: EnvironmentProtocol,
        history: ContactHistoryBuffer,
        device: str = "cpu",
        num_denoise_steps: int = 10,
    ):
        self.policy = policy
        self.environment = environment
        self.history = history
        self.device = device
        self.num_denoise_steps = int(num_denoise_steps)

    def run(self, *, max_steps: int = DEFAULT_MAX_STEPS) -> RolloutResult:
        steps: list[RolloutStep] = []
        for _ in range(max_steps):
            if self.environment.done():
                break
            now_s = float(self.environment.now_s)
            # 1) t 时刻可见的历史（只含已完成接触）。
            snapshot = self.history.snapshot(now_s)
            visible_contacts = int(snapshot["contact_mask"].sum())
            # 2) 构建 Observation 并推理。
            observation = self.environment.build_observation(snapshot)
            actions, diagnostics = self.policy.sample_actions(
                self.device,
                observation,
                num_steps=self.num_denoise_steps,
                return_diagnostics=True,
            )
            skill, selected_index = _diagnostic_skill_and_index(diagnostics)
            # 3) 执行；只有执行结束后返回的接触才写入记忆（下一次 snapshot 才可见）。
            outcome = self.environment.execute(actions, diagnostics)
            recorded = outcome is not None
            if recorded:
                self.history.add_completed_contact(
                    outcome.signal,
                    outcome.xyz_obj,
                    outcome.normal_obj,
                    outcome.end_time_s,
                )
            steps.append(
                RolloutStep(
                    now_s=now_s,
                    skill=skill,
                    selected_surface_index=selected_index,
                    visible_contacts=visible_contacts,
                    recorded_contact=recorded,
                )
            )
        return RolloutResult(steps=steps)


__all__ = [
    "DEFAULT_MAX_STEPS",
    "CompletedContact",
    "EnvironmentProtocol",
    "PolicyProtocol",
    "RolloutResult",
    "RolloutStep",
    "ThermalClosedLoop",
]

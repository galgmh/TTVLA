"""使用模型内部ThermalMemoryBank的在线闭环驱动。"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
from typing import Any, Protocol

import numpy as np

from training.pi05.thermal.labels import SKILL_GRASP, SKILL_PROBE

DEFAULT_MAX_STEPS = 64


class PolicyProtocol(Protocol):
    def reset_thermal_memory(self, batch_size: int | None = None) -> None: ...
    def sample_actions(self, device, observation, num_steps=10, *, return_diagnostics=True): ...


class EnvironmentProtocol(Protocol):
    @property
    def now_s(self) -> float: ...
    def build_observation(self): ...
    def execute(self, actions) -> None: ...
    def done(self) -> bool: ...


@dataclasses.dataclass(frozen=True)
class RolloutStep:
    now_s: float
    skill: int


@dataclasses.dataclass
class RolloutResult:
    steps: list[RolloutStep]

    @property
    def probe_steps(self) -> int:
        return sum(step.skill == SKILL_PROBE for step in self.steps)

    @property
    def grasp_steps(self) -> int:
        return sum(step.skill == SKILL_GRASP for step in self.steps)


def _diagnostic_skill(diagnostics: Mapping[str, Any]) -> int:
    probability = diagnostics.get("skill_probability")
    if probability is None:
        return -1
    first = probability[0]
    return int(first.detach().argmax()) if hasattr(first, "detach") else int(np.argmax(first))


class ThermalClosedLoop:
    """每个episode重置内部记忆；每步Observation只含当前传感器窗口。"""

    def __init__(self, *, policy: PolicyProtocol, environment: EnvironmentProtocol,
                 device: str = "cpu", num_denoise_steps: int = 10):
        self.policy = policy
        self.environment = environment
        self.device = device
        self.num_denoise_steps = num_denoise_steps

    def run(self, *, max_steps: int = DEFAULT_MAX_STEPS) -> RolloutResult:
        self.policy.reset_thermal_memory()
        steps = []
        for _ in range(max_steps):
            if self.environment.done():
                break
            observation = self.environment.build_observation()
            actions, diagnostics = self.policy.sample_actions(
                self.device, observation, num_steps=self.num_denoise_steps,
                return_diagnostics=True,
            )
            skill = _diagnostic_skill(diagnostics)
            now_s = float(self.environment.now_s)
            self.environment.execute(actions)
            steps.append(RolloutStep(now_s=now_s, skill=skill))
        return RolloutResult(steps)


__all__ = ["DEFAULT_MAX_STEPS", "EnvironmentProtocol", "PolicyProtocol", "RolloutResult", "RolloutStep", "ThermalClosedLoop"]

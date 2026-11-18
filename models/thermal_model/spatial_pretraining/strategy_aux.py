"""训练期全局技能辅助头。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

SKILL_PROBE = 0
SKILL_GRASP = 1
SKILL_RETRACT = 2
SKILL_ABORT = 3
NUM_SKILLS = 4


@dataclass
class StrategyAuxOutput:
    skill_logits: torch.Tensor


class GlobalSkillAuxHead(nn.Module):
    """从热记忆、图像、语言和状态预测四类技能。"""

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(4 * hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.skill_head = nn.Linear(hidden_dim, NUM_SKILLS)

    def forward(
        self,
        spatial_latents: torch.Tensor,
        image_summary: torch.Tensor,
        language_summary: torch.Tensor,
        robot_state: torch.Tensor,
    ) -> StrategyAuxOutput:
        thermal_summary = spatial_latents.mean(dim=1)
        fused = self.fusion(
            torch.cat(
                [thermal_summary, image_summary, language_summary, robot_state],
                dim=-1,
            )
        )
        return StrategyAuxOutput(skill_logits=self.skill_head(fused))


__all__ = [
    "GlobalSkillAuxHead",
    "NUM_SKILLS",
    "SKILL_ABORT",
    "SKILL_GRASP",
    "SKILL_PROBE",
    "SKILL_RETRACT",
    "StrategyAuxOutput",
]

"""全局技能辅助头：不预测候选点，也不直接控制动作。"""

from __future__ import annotations

import torch
from torch import nn

from models.pi05.thermal.outputs import StrategyAuxOutput

SKILL_PROBE = 0
SKILL_GRASP = 1
SKILL_RETRACT = 2
SKILL_ABORT = 3
NUM_SKILLS = 4


class GlobalSkillAuxHead(nn.Module):
    """从全局热记忆和任务上下文预测PROBE/GRASP/RETRACT/ABORT。"""

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
        fused = self.fusion(torch.cat([
            thermal_summary,
            image_summary,
            language_summary,
            robot_state,
        ], dim=-1))
        skill_logits = self.skill_head(fused)
        return StrategyAuxOutput(skill_logits=skill_logits)


__all__ = [
    "GlobalSkillAuxHead", "NUM_SKILLS", "SKILL_ABORT", "SKILL_GRASP", "SKILL_PROBE", "SKILL_RETRACT",
]

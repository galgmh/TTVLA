"""主动策略：技能分类（OTHER/PROBE/GRASP）+ 可微目标点选择 + 目标特征。"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

from models.pi05.thermal.outputs import StrategyOutput


def _safe_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """把非法点设为 dtype 最小值；整行都非法时退回全零（softmax 为均匀分布）。"""
    neg = torch.finfo(logits.dtype).min
    masked = torch.where(mask, logits, torch.full_like(logits, neg))
    any_valid = mask.any(dim=-1, keepdim=True)
    return torch.where(any_valid, masked, torch.zeros_like(masked))


def _select_distribution(
    logits: torch.Tensor,
    mask: torch.Tensor,
    mode: str,
    tau: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    safe = _safe_logits(logits, mask)
    if mode == "argmax":
        index = safe.argmax(dim=-1)
        return F.one_hot(index, safe.shape[-1]).to(safe.dtype), index
    if mode == "gumbel":
        # straight-through：前向是 one-hot，反向有 soft 梯度。
        return F.gumbel_softmax(safe, tau=tau, hard=True, dim=-1), None
    return torch.softmax(safe, dim=-1), None


class ActiveStrategyModule(nn.Module):
    """决定下一步技能与目标表面点，并输出可微的 goal feature。

    训练前期用 softmax 分布；训练后期可切换 straight-through Gumbel；
    eval（self.training=False）时强制 masked argmax。
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.0,
        min_grasp_score: float = 0.3,
        gumbel_tau: float = 1.0,
        selection_mode: str = "soft",
        num_skills: int = 3,
    ):
        super().__init__()
        self.num_skills = num_skills
        self.min_grasp_score = min_grasp_score
        self.gumbel_tau = gumbel_tau
        self.selection_mode = selection_mode
        # 全局上下文：图像摘要 + 语言摘要 + 机器人状态 + 剩余 probe 预算。
        # 预算以原始标量 [B,1] 输入，模块内部投影并保留原始值用于 mask。
        self.budget_proj = nn.Linear(1, hidden_dim)
        self.global_mlp = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.skill_head = nn.Linear(hidden_dim, num_skills)
        self.probe_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.grasp_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.skill_embeddings = nn.Parameter(torch.zeros(num_skills, hidden_dim))
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        surface_features: torch.Tensor,
        surface_mask: torch.Tensor,
        surface_reachable: torch.Tensor,
        surface_grasp_score: torch.Tensor,
        image_summary: torch.Tensor,
        language_summary: torch.Tensor,
        robot_state: torch.Tensor,
        probe_budget: torch.Tensor,
        mode: str | None = None,
    ) -> StrategyOutput:
        b, n, _ = surface_features.shape
        valid = surface_mask.bool()
        reachable = valid & surface_reachable.bool()
        grasp_ok = reachable & (surface_grasp_score.squeeze(-1) >= self.min_grasp_score)

        global_context = self.global_mlp(
            torch.cat([image_summary, language_summary, robot_state, self.budget_proj(probe_budget)], dim=-1)
        )
        pooled = (surface_features * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1)
        skill_logits = self.skill_head(self.output_norm(global_context + pooled))

        # budget 为 0 时禁止 PROBE 技能（把该 logit 压到 dtype 最小值）。
        no_budget = (probe_budget.squeeze(-1) <= 0).unsqueeze(-1)
        if no_budget.any():
            neg = torch.finfo(skill_logits.dtype).min
            is_probe = torch.zeros(self.num_skills, dtype=torch.bool, device=skill_logits.device)
            is_probe[1] = True
            blocked = no_budget & is_probe
            skill_logits = torch.where(blocked, torch.full_like(skill_logits, neg), skill_logits)

        point_context = global_context.unsqueeze(1).expand(-1, n, -1)
        point_input = torch.cat([surface_features, point_context], dim=-1)
        probe_logits = self.probe_head(point_input).squeeze(-1)
        grasp_logits = self.grasp_head(point_input).squeeze(-1)
        # 返回的 logits 本身就带 mask（CE 监督同样只看合法点）。
        neg = torch.finfo(probe_logits.dtype).min
        probe_logits = torch.where(reachable, probe_logits, torch.full_like(probe_logits, neg))
        grasp_logits = torch.where(grasp_ok, grasp_logits, torch.full_like(grasp_logits, neg))

        if mode is None:
            mode = self.selection_mode if self.training else "argmax"
        probe_dist, _ = _select_distribution(probe_logits, reachable, mode, self.gumbel_tau)
        grasp_dist, _ = _select_distribution(grasp_logits, grasp_ok, mode, self.gumbel_tau)
        skill_probs = torch.softmax(skill_logits, dim=-1)

        p_probe = skill_probs[:, 1:2]
        p_grasp = skill_probs[:, 2:3]
        selected = p_probe * probe_dist + p_grasp * grasp_dist
        probe_goal = torch.einsum("bn,bnh->bh", probe_dist, surface_features)
        grasp_goal = torch.einsum("bn,bnh->bh", grasp_dist, surface_features)
        skill_goal = torch.einsum("bs,sh->bh", skill_probs, self.skill_embeddings)
        goal_feature = p_probe * probe_goal + p_grasp * grasp_goal + skill_goal

        selected_index = selected.argmax(dim=-1) if mode == "argmax" else None
        return StrategyOutput(
            skill_logits=skill_logits,
            probe_logits=probe_logits,
            grasp_logits=grasp_logits,
            selected_distribution=selected,
            goal_feature=goal_feature,
            selected_index=selected_index,
        )

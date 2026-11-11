"""IntegratedThermalEncoder：把热觉历史/候选点/策略输出接成 π0.5 prefix token。

输出 ``prefix_tokens`` 形状 [B, M+1, paligemma_width]，由 π0.5 的
``embed_prefix`` 追加到图像与语言 token 之后；本模块不接触 noisy action
suffix，也不修改 adaRMS 时间条件路径。
"""

from __future__ import annotations

import torch
from torch import nn

from models.pi05.thermal.config import ThermalPi0Config
from models.pi05.thermal.contact_encoder import ThermalContactEncoder
from models.pi05.thermal.outputs import IntegratedThermalOutput
from models.pi05.thermal.point_encoder import SurfacePointEncoder
from models.pi05.thermal.spatial_memory import ContactTokenBuilder
from models.pi05.thermal.spatial_memory import ThermalSpatialMemory
from models.pi05.thermal.strategy import ActiveStrategyModule


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """x: [B, T, D]；mask: [B, T] bool -> [B, D]。"""
    w = mask.to(x.dtype).unsqueeze(-1)
    return (x * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)


class IntegratedThermalEncoder(nn.Module):
    """热觉分支总装。所有维度来自 ``config``，宽度来自 π0.5 的 paligemma 配置。"""

    def __init__(
        self,
        config: ThermalPi0Config,
        paligemma_width: int,
        state_dim: int = 32,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.config = config
        self.contact_encoder = ThermalContactEncoder(
            signal_dim=config.signal_dim,
            latent_dim=config.thermal_latent_dim,
        )
        self.contact_builder = ContactTokenBuilder(
            latent_dim=config.thermal_latent_dim,
            hidden_dim=config.hidden_dim,
        )
        self.point_encoder = SurfacePointEncoder(hidden_dim=config.hidden_dim)
        self.spatial_memory = ThermalSpatialMemory(
            hidden_dim=config.hidden_dim,
            num_spatial_tokens=config.num_spatial_tokens,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.strategy = ActiveStrategyModule(
            hidden_dim=config.hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            min_grasp_score=config.min_grasp_score,
            gumbel_tau=config.strategy_gumbel_tau,
            selection_mode=config.selection_mode,
        )
        # 把 π0.5 的视觉/语言摘要投影到热觉模块宽度，再投影回 paligemma 宽度。
        self.image_proj = nn.Linear(paligemma_width, config.hidden_dim)
        self.language_proj = nn.Linear(paligemma_width, config.hidden_dim)
        self.state_proj = nn.Linear(state_dim, config.hidden_dim)
        self.spatial_proj = nn.Linear(config.hidden_dim, paligemma_width)
        self.goal_proj = nn.Linear(config.hidden_dim, paligemma_width)
        # 两类新增 token 的可学习类型嵌入：0=空间记忆, 1=主动目标。
        self.type_embedding = nn.Parameter(torch.zeros(2, 1, paligemma_width))

    def _get(self, observation, name: str, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        value = getattr(observation, name, None)
        if value is None:
            raise ValueError(f"thermal_enabled=True 时 observation.{name} 不能为 None")
        if value.is_floating_point():
            return value.to(device=device, dtype=dtype)
        return value.to(device=device)

    def _validate_shapes(self, observation) -> None:
        cfg = self.config
        required = (
            "thermal_history",
            "thermal_history_time_mask",
            "contact_mask",
            "contact_xyz_obj",
            "contact_normal_obj",
            "contact_age_s",
            "surface_xyz_obj",
            "surface_normal_obj",
            "surface_mask",
            "surface_reachable",
            "surface_grasp_score",
            "ee_xyz_obj",
            "probe_budget_remaining",
        )
        missing = [name for name in required if getattr(observation, name, None) is None]
        if missing:
            raise ValueError(f"thermal_enabled=True 时 observation 缺少热觉字段: {missing}")

        history = observation.thermal_history
        if history.dim() != 4:
            raise ValueError(f"thermal_history 必须是 [B,K,L,C]，得到 {tuple(history.shape)}")
        b, k, length, channels = history.shape
        if k != cfg.history_size or length != cfg.window_size or channels != cfg.signal_dim:
            raise ValueError(
                "thermal_history 形状与 ThermalPi0Config 不一致："
                f"得到 K={k}, L={length}, C={channels}；期望 K={cfg.history_size}, "
                f"L={cfg.window_size}, C={cfg.signal_dim}"
            )
        if observation.surface_xyz_obj.shape[1] != cfg.num_surface_points:
            raise ValueError(
                f"surface_xyz_obj 的 N={observation.surface_xyz_obj.shape[1]}，"
                f"期望 {cfg.num_surface_points}"
            )

    def forward(
        self,
        observation,
        image_embeddings: list[torch.Tensor],
        language_embeddings: torch.Tensor,
    ) -> IntegratedThermalOutput:
        cfg = self.config
        param = next(self.parameters())
        dtype, device = param.dtype, param.device
        self._validate_shapes(observation)

        history = self._get(observation, "thermal_history", dtype, device)
        time_mask = self._get(observation, "thermal_history_time_mask", dtype, device).bool()
        contact_mask = self._get(observation, "contact_mask", dtype, device).bool()

        # --- π0.5 视觉/语言摘要 ---
        image_masks = getattr(observation, "image_masks", None)
        summaries, weights = [], []
        for i, emb in enumerate(image_embeddings):
            if image_masks is None:
                weight = emb.new_ones(emb.shape[0], 1)
            else:
                weight = list(image_masks.values())[i].to(device=device, dtype=dtype).view(-1, 1)
            summaries.append(emb.to(dtype).mean(dim=1) * weight)
            weights.append(weight)
        if summaries:
            image_summary = torch.stack(summaries).sum(0) / torch.stack(weights).sum(0).clamp_min(1.0)
        else:
            image_summary = param.new_zeros(history.shape[0], cfg.hidden_dim)

        lang = language_embeddings.to(dtype=dtype, device=device)
        lang_mask = getattr(observation, "tokenized_prompt_mask", None)
        lang_summary = (
            lang.mean(dim=1) if lang_mask is None else _masked_mean(lang, lang_mask.to(device=device).bool())
        )
        lang_summary = self.language_proj(lang_summary)

        # --- 局部接触编码 -> contact token ---
        contact = self.contact_encoder(history, time_mask, contact_mask)
        contact_tokens = self.contact_builder(
            contact.latent,
            self._get(observation, "contact_xyz_obj", dtype, device),
            self._get(observation, "contact_normal_obj", dtype, device),
            contact.temp_mean_c,
            contact.temp_logvar,
            torch.sigmoid(contact.valid_logit),
            self._get(observation, "contact_age_s", dtype, device),
        )

        # --- 候选表面点编码 -> 空间记忆 ---
        surface_features = self.point_encoder(
            self._get(observation, "surface_xyz_obj", dtype, device),
            self._get(observation, "surface_normal_obj", dtype, device),
            self._get(observation, "surface_reachable", dtype, device),
            self._get(observation, "surface_grasp_score", dtype, device),
            self._get(observation, "ee_xyz_obj", dtype, device),
        )
        memory = self.spatial_memory(
            contact_tokens,
            contact_mask,
            surface_features,
            self._get(observation, "surface_mask", dtype, device).bool(),
            lang_summary,
        )

        # --- 主动策略 ---
        strategy = self.strategy(
            surface_features=memory.surface_features,
            surface_mask=self._get(observation, "surface_mask", dtype, device).bool(),
            surface_reachable=self._get(observation, "surface_reachable", dtype, device).bool(),
            surface_grasp_score=self._get(observation, "surface_grasp_score", dtype, device),
            image_summary=self.image_proj(image_summary),
            language_summary=lang_summary,
            robot_state=self.state_proj(self._get(observation, "state", dtype, device)),
            probe_budget=self._get(observation, "probe_budget_remaining", dtype, device),
        )

        # --- 压缩成 π0.5 prefix token：M 个空间记忆 token + 1 个主动目标 token ---
        spatial_tokens = self.spatial_proj(memory.spatial_latents) + self.type_embedding[0]
        goal_token = self.goal_proj(strategy.goal_feature).unsqueeze(1) + self.type_embedding[1]
        prefix_tokens = torch.cat([spatial_tokens, goal_token], dim=1)
        prefix_mask = torch.ones(prefix_tokens.shape[:2], dtype=torch.bool, device=device)

        return IntegratedThermalOutput(
            prefix_tokens=prefix_tokens,
            prefix_mask=prefix_mask,
            contact=contact,
            field_temp_mean_c=memory.temp_mean_c,
            field_temp_logvar=memory.temp_logvar,
            field_safe_logit=memory.safe_logit,
            strategy=strategy,
        )

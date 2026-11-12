"""在线Thermal-π0.5：当前滑动窗口编码、事件记忆和prefix注入。"""

from __future__ import annotations

import torch
from torch import nn

from models.pi05.thermal.config import ThermalPi0Config
from models.pi05.thermal.contact_encoder import ThermalContactEncoder
from models.pi05.thermal.field_decoder import SpatialFieldDecoder
from models.pi05.thermal.memory_bank import ThermalMemoryBank
from models.pi05.thermal.outputs import IntegratedThermalOutput
from models.pi05.thermal.prefix_adapter import ThermalPrefixAdapter
from models.pi05.thermal.spatial_memory import ContactTokenBuilder, ThermalSpatialMemory
from models.pi05.thermal.strategy_aux import GlobalSkillAuxHead


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(x.dtype).unsqueeze(-1)
    return (x * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)


class IntegratedThermalEncoder(nn.Module):
    """把当前热触觉窗口和内部完成接触记忆压成M个prefix token。"""

    def __init__(self, config: ThermalPi0Config, paligemma_width: int, state_dim: int = 32,
                 num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.config = config
        self.paligemma_width = paligemma_width
        self.contact_encoder = ThermalContactEncoder(
            signal_dim=config.signal_dim, latent_dim=config.thermal_latent_dim
        )
        self.contact_builder = ContactTokenBuilder(
            latent_dim=config.thermal_latent_dim, hidden_dim=config.hidden_dim
        )
        self.spatial_memory = ThermalSpatialMemory(
            hidden_dim=config.hidden_dim, num_spatial_tokens=config.num_spatial_tokens,
            num_heads=num_heads, dropout=dropout,
        )
        self.prefix_adapter = ThermalPrefixAdapter(
            hidden_dim=config.hidden_dim, paligemma_width=paligemma_width,
            adapter_hidden_dim=config.prefix_adapter_hidden_dim, gate_init=config.prefix_gate_init,
        )
        self.field_decoder = SpatialFieldDecoder(
            hidden_dim=config.hidden_dim, bands=config.field_fourier_bands, num_heads=num_heads
        )
        self.strategy_aux = GlobalSkillAuxHead(hidden_dim=config.hidden_dim)
        self.image_proj = nn.Linear(paligemma_width, config.hidden_dim)
        self.language_proj = nn.Linear(paligemma_width, config.hidden_dim)
        self.state_proj = nn.Linear(state_dim, config.hidden_dim)
        self.memory_bank = ThermalMemoryBank(
            config.memory_size, config.hidden_dim,
            on_threshold=config.contact_on_threshold,
            off_threshold=config.contact_off_threshold,
            on_steps=config.contact_on_steps,
            off_steps=config.contact_off_steps,
            momentum=config.active_token_momentum,
        )

    def reset_memory(self, batch_size: int | None = None, *, device=None, dtype=None) -> None:
        """新episode开始时调用；不传batch_size则完全清除，下一次前向自动初始化。"""
        if batch_size is None:
            self.memory_bank.clear()
            return
        parameter = next(self.parameters())
        self.memory_bank.reset(
            batch_size, device=parameter.device if device is None else torch.device(device),
            dtype=parameter.dtype if dtype is None else dtype,
        )

    def _get(self, observation, name: str, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        value = getattr(observation, name, None)
        if value is None:
            raise ValueError(f"thermal_enabled=True时observation.{name}不能为None")
        return value.to(device=device, dtype=dtype) if value.is_floating_point() else value.to(device=device)

    def _validate_shapes(self, observation) -> None:
        required = (
            "thermal_window", "thermal_window_time_mask", "sensor_xyz_base",
            "sensor_normal_base",
        )
        missing = [name for name in required if getattr(observation, name, None) is None]
        if missing:
            raise ValueError(f"thermal_enabled=True时observation缺少字段: {missing}")
        window = observation.thermal_window
        expected = (self.config.window_size, self.config.signal_dim)
        if window.dim() != 3 or tuple(window.shape[1:]) != expected:
            raise ValueError(f"thermal_window必须为[B,L,C]且L/C={expected}，得到{tuple(window.shape)}")
        if observation.thermal_window_time_mask.shape != window.shape[:2]:
            raise ValueError("thermal_window_time_mask必须为[B,L]")

    def _encode_current(self, observation):
        parameter = next(self.parameters())
        dtype, device = parameter.dtype, parameter.device
        window = self._get(observation, "thermal_window", dtype, device)
        time_mask = self._get(observation, "thermal_window_time_mask", dtype, device).bool()
        contact = self.contact_encoder(window, time_mask)
        normal = torch.nn.functional.normalize(
            self._get(observation, "sensor_normal_base", dtype, device), dim=-1, eps=1e-6
        )
        token = self.contact_builder(
            contact.latent,
            self._get(observation, "sensor_xyz_base", dtype, device),
            normal,
            contact.temp_mean_c,
            contact.temp_logvar,
            torch.sigmoid(contact.valid_logit),
        )
        return contact, token, time_mask.any(dim=1)

    def _memory_input(self, token, contact, window_present, *, update_memory: bool):
        probability = torch.sigmoid(contact.valid_logit)
        if update_memory:
            self.memory_bank.update(token, probability)
            return self.memory_bank.read_with_current(token, probability)
        # 训练batch默认相互独立：只使用当前临时token，不持久化跨样本状态。
        return token[:, None], window_present[:, None]

    def forward_physical(self, observation) -> IntegratedThermalOutput:
        """Stage 1单窗口热物理路径，不运行视觉、语言、prefix或在线持久记忆。"""
        self._validate_shapes(observation)
        parameter = next(self.parameters())
        contact, token, present = self._encode_current(observation)
        memory = self.spatial_memory(token[:, None], present[:, None])
        batch = token.shape[0]
        return IntegratedThermalOutput(
            prefix_tokens=parameter.new_zeros((batch, 0, self.paligemma_width)),
            prefix_mask=torch.zeros((batch, 0), dtype=torch.bool, device=parameter.device),
            contact=contact, spatial_latents=memory.spatial_latents, field=None, strategy=None,
        )

    def decode_field(self, output: IntegratedThermalOutput, query_xyz_base: torch.Tensor) -> None:
        parameter = next(self.parameters())
        output.field = self.field_decoder(
            output.spatial_latents,
            query_xyz_base.to(device=parameter.device, dtype=parameter.dtype),
        )

    def forward(self, observation, image_embeddings: list[torch.Tensor],
                language_embeddings: torch.Tensor, *, update_memory: bool = False) -> IntegratedThermalOutput:
        self._validate_shapes(observation)
        parameter = next(self.parameters())
        dtype, device = parameter.dtype, parameter.device
        batch = observation.thermal_window.shape[0]

        image_masks = getattr(observation, "image_masks", None)
        summaries, weights = [], []
        for index, embedding in enumerate(image_embeddings):
            embedding = embedding.to(device=device, dtype=dtype)
            weight = embedding.new_ones((batch, 1)) if image_masks is None else (
                list(image_masks.values())[index].to(device=device, dtype=dtype).view(-1, 1)
            )
            summaries.append(embedding.mean(dim=1) * weight)
            weights.append(weight)
        image_base = (
            torch.stack(summaries).sum(0) / torch.stack(weights).sum(0).clamp_min(1.0)
            if summaries else parameter.new_zeros((batch, self.paligemma_width))
        )
        image_summary = self.image_proj(image_base)

        language = language_embeddings.to(device=device, dtype=dtype)
        language_mask = getattr(observation, "tokenized_prompt_mask", None)
        language_base = language.mean(dim=1) if language_mask is None else _masked_mean(
            language, language_mask.to(device=device).bool()
        )

        contact, current_token, present = self._encode_current(observation)
        tokens, token_mask = self._memory_input(
            current_token, contact, present, update_memory=update_memory
        )
        memory = self.spatial_memory(tokens, token_mask)
        prefix_tokens = self.prefix_adapter(memory.spatial_latents)
        prefix_mask = torch.ones(prefix_tokens.shape[:2], dtype=torch.bool, device=device)
        strategy = self.strategy_aux(
            spatial_latents=memory.spatial_latents,
            image_summary=image_summary,
            language_summary=self.language_proj(language_base),
            robot_state=self.state_proj(self._get(observation, "state", dtype, device)),
        )
        return IntegratedThermalOutput(
            prefix_tokens=prefix_tokens, prefix_mask=prefix_mask, contact=contact,
            spatial_latents=memory.spatial_latents, field=None, strategy=strategy,
        )


__all__ = ["IntegratedThermalEncoder"]

"""在线Thermal-π0.5：当前滑动窗口编码、事件记忆和prefix注入。"""

from __future__ import annotations

import torch
from torch import nn

from thermal_model.config import ThermalPi0Config
from thermal_model.contact_encoder import ThermalContactEncoder
from thermal_model.memory_bank import ThermalMemoryBank
from thermal_model.observation import IntegratedThermalOutput
from thermal_model.prefix_adapter import ThermalPrefixAdapter
from thermal_model.preprocessing import FingertipGeometryProvider
from thermal_model.spatial_memory import ContactTokenBuilder, ThermalSpatialMemory


class IntegratedThermalEncoder(nn.Module):
    """把当前热触觉窗口和内部完成接触记忆压成M个prefix token。"""

    def __init__(
        self,
        config: ThermalPi0Config,
        paligemma_width: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        fingertip_geometry: FingertipGeometryProvider | None = None,
    ):
        super().__init__()
        self.config = config
        self.paligemma_width = paligemma_width
        self.fingertip_geometry = fingertip_geometry
        self.contact_encoder = ThermalContactEncoder(
            signal_dim=config.signal_dim,
            latent_dim=config.thermal_latent_dim,
            logvar_min=config.contact_logvar_min,
            logvar_max=config.contact_logvar_max,
        )
        self.contact_builder = ContactTokenBuilder(
            latent_dim=config.thermal_latent_dim,
            hidden_dim=config.hidden_dim,
            bands=config.field_fourier_bands,
        )
        self.spatial_memory = ThermalSpatialMemory(
            hidden_dim=config.hidden_dim, num_spatial_tokens=config.num_spatial_tokens,
            num_heads=num_heads, dropout=dropout,
        )
        self.prefix_adapter = ThermalPrefixAdapter(
            hidden_dim=config.hidden_dim, paligemma_width=paligemma_width,
            adapter_hidden_dim=config.prefix_adapter_hidden_dim, gate_init=config.prefix_gate_init,
        )
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
        required = ("thermal_window", "thermal_window_time_mask", "state")
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
        if self.fingertip_geometry is None:
            raise ValueError(
                "IntegratedThermalEncoder needs fingertip_geometry to map state[..., :8] to pad poses"
            )
        state = self._get(observation, "state", dtype, device)[..., :8]
        geometry = self.fingertip_geometry(state)
        xyz = geometry.positions_base.to(device=device, dtype=dtype)
        normal = torch.nn.functional.normalize(
            geometry.normals_base.to(device=device, dtype=dtype), dim=-1, eps=1e-6
        )
        token = self.contact_builder(
            contact.latent,
            xyz,
            normal,
        )
        return contact, token, time_mask.any(dim=1)

    def _memory_input(self, token, contact, window_present, *, update_memory: bool):
        probability = torch.sigmoid(contact.valid_logit)
        if update_memory:
            self.memory_bank.update(token, probability)
            return self.memory_bank.read_with_current(token, probability)
        # 训练batch默认相互独立：只使用当前临时token，不持久化跨样本状态。
        return token[:, None], window_present[:, None]

    def forward(self, observation, *, update_memory: bool = False) -> IntegratedThermalOutput:
        """把当前热觉记忆转换成供 π0.5 使用的 thermal prefix。"""
        self._validate_shapes(observation)
        parameter = next(self.parameters())
        dtype, device = parameter.dtype, parameter.device

        contact, current_token, present = self._encode_current(observation)
        tokens, token_mask = self._memory_input(
            current_token, contact, present, update_memory=update_memory
        )
        memory = self.spatial_memory(tokens, token_mask)
        prefix_tokens = self.prefix_adapter(memory.spatial_latents)
        prefix_mask = torch.ones(prefix_tokens.shape[:2], dtype=torch.bool, device=device)
        return IntegratedThermalOutput(
            prefix_tokens=prefix_tokens, prefix_mask=prefix_mask, contact=contact,
            spatial_latents=memory.spatial_latents,
        )


__all__ = ["IntegratedThermalEncoder"]

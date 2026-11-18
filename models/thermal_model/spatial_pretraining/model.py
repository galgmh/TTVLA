"""冻结接触编码器之后的接触空间表示预训练模型。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from thermal_model.contact_encoder import ThermalContactEncoder
from thermal_model.observation import ContactEncoderOutput
from thermal_model.spatial_memory import ContactTokenBuilder, ThermalSpatialMemory
from thermal_model.spatial_pretraining.field_decoder import (
    FieldDecoderOutput,
    SpatialFieldDecoder,
)


@dataclass
class SpatialPretrainOutput:
    contact: ContactEncoderOutput
    spatial_latents: torch.Tensor
    field: FieldDecoderOutput


class SpatialPretrainModel(nn.Module):
    """接触窗口 + 指垫几何 -> 空间热场查询。"""

    def __init__(
        self,
        *,
        signal_dim: int,
        contact_hidden_channels: tuple[int, ...],
        contact_dilations: tuple[int, ...],
        contact_kernel_size: int,
        thermal_latent_dim: int,
        contact_logvar_min: float,
        contact_logvar_max: float,
        hidden_dim: int,
        num_spatial_tokens: int,
        num_heads: int,
        fourier_bands: int,
        field_logvar_min: float,
        field_logvar_max: float,
        dropout: float,
    ):
        super().__init__()
        self.contact_encoder = ThermalContactEncoder(
            signal_dim=signal_dim,
            hidden_channels=contact_hidden_channels,
            dilations=contact_dilations,
            kernel_size=contact_kernel_size,
            latent_dim=thermal_latent_dim,
            logvar_min=contact_logvar_min,
            logvar_max=contact_logvar_max,
        )
        self.contact_builder = ContactTokenBuilder(
            latent_dim=thermal_latent_dim,
            hidden_dim=hidden_dim,
            bands=fourier_bands,
        )
        self.spatial_memory = ThermalSpatialMemory(
            hidden_dim=hidden_dim,
            num_spatial_tokens=num_spatial_tokens,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.field_decoder = SpatialFieldDecoder(
            hidden_dim=hidden_dim,
            bands=fourier_bands,
            num_heads=num_heads,
            logvar_min=field_logvar_min,
            logvar_max=field_logvar_max,
        )

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        positions: torch.Tensor,
        normals: torch.Tensor,
        query_xyz: torch.Tensor,
    ) -> SpatialPretrainOutput:
        contact = self.contact_encoder(features, mask)
        token = self.contact_builder(contact.latent, positions, normals)
        present = mask.any(dim=1)
        spatial = self.spatial_memory(token[:, None], present[:, None])
        field = self.field_decoder(spatial.spatial_latents, query_xyz)
        return SpatialPretrainOutput(
            contact=contact,
            spatial_latents=spatial.spatial_latents,
            field=field,
        )


__all__ = ["SpatialPretrainModel", "SpatialPretrainOutput"]

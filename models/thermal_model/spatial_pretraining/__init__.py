"""Spatial thermal representation pretraining."""

from thermal_model.spatial_pretraining.config import build_model, load_model_config
from thermal_model.spatial_pretraining.field_decoder import (
    FieldDecoderOutput,
    SpatialFieldDecoder,
)
from thermal_model.spatial_pretraining.losses import compute_losses
from thermal_model.spatial_pretraining.model import SpatialPretrainModel
from thermal_model.spatial_pretraining.strategy_aux import (
    GlobalSkillAuxHead,
    StrategyAuxOutput,
)

__all__ = [
    "FieldDecoderOutput",
    "GlobalSkillAuxHead",
    "SpatialPretrainModel",
    "SpatialFieldDecoder",
    "StrategyAuxOutput",
    "build_model",
    "compute_losses",
    "load_model_config",
]

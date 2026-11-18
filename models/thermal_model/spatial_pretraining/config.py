"""Configuration for standalone spatial thermal pretraining."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

from sim.config import REPO_ROOT
from thermal_model.spatial_pretraining.model import SpatialPretrainModel

_MODEL_KEYS = {
    "signal_dim",
    "contact_hidden_channels",
    "contact_dilations",
    "contact_kernel_size",
    "thermal_latent_dim",
    "contact_logvar_min",
    "contact_logvar_max",
    "hidden_dim",
    "num_spatial_tokens",
    "num_heads",
    "fourier_bands",
    "field_logvar_min",
    "field_logvar_max",
    "dropout",
}
_TRAINING_KEYS = {
    "encoder_checkpoint",
    "freeze_encoder",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "epochs",
    "seed",
    "field_loss",
    "huber_delta",
    "weights",
    "output_dir",
    "device",
    "num_workers",
}
_DATA_KEYS = {"root", "train", "validation", "test"}
_WEIGHT_KEYS = {"field_temp"}


def _check_keys(mapping: Any, required: set[str], where: str) -> None:
    if not isinstance(mapping, dict):
        raise ValueError(f"{where} must be a mapping")
    missing = required - set(mapping)
    unknown = set(mapping) - required
    if missing:
        raise ValueError(f"{where} is missing keys: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{where} has unknown keys: {sorted(unknown)}")


def load_model_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    _check_keys(cfg, {"model", "training", "data"}, str(path))
    _check_keys(cfg["model"], _MODEL_KEYS, f"{path}: model")
    _check_keys(cfg["training"], _TRAINING_KEYS, f"{path}: training")
    _check_keys(cfg["training"]["weights"], _WEIGHT_KEYS, f"{path}: training.weights")
    _check_keys(cfg["data"], _DATA_KEYS, f"{path}: data")
    model = cfg["model"]
    if len(model["contact_hidden_channels"]) != len(model["contact_dilations"]):
        raise ValueError("contact_hidden_channels 与 contact_dilations 长度必须一致")
    return cfg


def build_model(cfg: dict[str, Any]) -> SpatialPretrainModel:
    model = cfg["model"]
    return SpatialPretrainModel(
        signal_dim=int(model["signal_dim"]),
        contact_hidden_channels=tuple(int(v) for v in model["contact_hidden_channels"]),
        contact_dilations=tuple(int(v) for v in model["contact_dilations"]),
        contact_kernel_size=int(model["contact_kernel_size"]),
        thermal_latent_dim=int(model["thermal_latent_dim"]),
        contact_logvar_min=float(model["contact_logvar_min"]),
        contact_logvar_max=float(model["contact_logvar_max"]),
        hidden_dim=int(model["hidden_dim"]),
        num_spatial_tokens=int(model["num_spatial_tokens"]),
        num_heads=int(model["num_heads"]),
        fourier_bands=int(model["fourier_bands"]),
        field_logvar_min=float(model["field_logvar_min"]),
        field_logvar_max=float(model["field_logvar_max"]),
        dropout=float(model["dropout"]),
    )


def load_encoder_checkpoint(model: SpatialPretrainModel, path: str | Path) -> Path:
    checkpoint = Path(path).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = REPO_ROOT / checkpoint
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.contact_encoder.load_state_dict(payload["model"])
    return checkpoint


__all__ = ["build_model", "load_encoder_checkpoint", "load_model_config"]

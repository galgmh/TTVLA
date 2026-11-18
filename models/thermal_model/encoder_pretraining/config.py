"""Configuration for standalone contact-encoder pretraining."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from thermal_model.contact_encoder import ThermalContactEncoder

_MODEL_KEYS = {
    "signal_dim",
    "hidden_channels",
    "dilations",
    "kernel_size",
    "latent_dim",
    "logvar_min",
    "logvar_max",
}
_TRAINING_KEYS = {
    "batch_size",
    "learning_rate",
    "weight_decay",
    "epochs",
    "seed",
    "weights",
    "output_dir",
    "device",
    "num_workers",
}
_DATA_KEYS = {
    "root",
    "train",
    "validation",
    "test",
}
_WEIGHT_KEYS = {"temp", "valid"}


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
    if len(cfg["model"]["hidden_channels"]) != len(cfg["model"]["dilations"]):
        raise ValueError("hidden_channels and dilations must have equal length")
    return cfg


def build_encoder(cfg: dict[str, Any]) -> ThermalContactEncoder:
    model = cfg["model"]
    return ThermalContactEncoder(
        signal_dim=int(model["signal_dim"]),
        hidden_channels=tuple(int(v) for v in model["hidden_channels"]),
        dilations=tuple(int(v) for v in model["dilations"]),
        kernel_size=int(model["kernel_size"]),
        latent_dim=int(model["latent_dim"]),
        logvar_min=float(model.get("logvar_min", -4.0)),
        logvar_max=float(model.get("logvar_max", 3.0)),
    )


__all__ = ["build_encoder", "load_model_config"]

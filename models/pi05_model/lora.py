"""Native PyTorch LoRA adapters for the local PI0/PI0.5 implementation.

This module has no PEFT/JAX dependency.  Base weights are loaded first, then
selected ``nn.Linear`` modules are replaced by :class:`LoRALinear` while
preserving their parameters exactly.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Literal

import safetensors.torch
import torch
from torch import nn
from torch.nn import functional as F


@dataclasses.dataclass(frozen=True)
class LoRAConfig:
    rank: int
    alpha: float
    dropout: float = 0.0
    rslora: bool = False
    init_std: float = 0.01

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("LoRA rank must be >= 1")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")

    @property
    def scale(self) -> float:
        denominator = math.sqrt(self.rank) if self.rslora else self.rank
        return float(self.alpha) / denominator


@dataclasses.dataclass(frozen=True)
class LoRATrainingConfig:
    enabled: bool = False
    paligemma_attn_rank: int = 16
    paligemma_ffn_rank: int = 16
    paligemma_attn_alpha: float = 16.0
    paligemma_ffn_alpha: float = 16.0
    expert_attn_rank: int = 32
    expert_ffn_rank: int = 32
    expert_attn_alpha: float = 32.0
    expert_ffn_alpha: float = 32.0
    dropout: float = 0.0
    use_rslora: bool = False
    apply_to: Literal[
        "all", "paligemma_only", "expert_only", "paligemma_attn", "expert_attn"
    ] = "all"
    train_vision_encoder: bool = False
    train_non_lora_layers: bool = True
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    trainable_modules: tuple[str, ...] = (
        "action_in_proj", "action_out_proj",
        "time_mlp_in", "time_mlp_out",
        "state_proj", "action_time_mlp_in", "action_time_mlp_out",
    )

    def __post_init__(self) -> None:
        ranks = (
            self.paligemma_attn_rank,
            self.paligemma_ffn_rank,
            self.expert_attn_rank,
            self.expert_ffn_rank,
        )
        if any(rank < 1 for rank in ranks):
            raise ValueError("All LoRA ranks must be >= 1")

    def configs(self) -> dict[str, dict[str, LoRAConfig]]:
        common = {"dropout": self.dropout, "rslora": self.use_rslora}
        return {
            "paligemma": {
                "attn": LoRAConfig(
                    self.paligemma_attn_rank,
                    self.paligemma_attn_alpha,
                    **common,
                ),
                "ffn": LoRAConfig(
                    self.paligemma_ffn_rank,
                    self.paligemma_ffn_alpha,
                    **common,
                ),
            },
            "expert": {
                "attn": LoRAConfig(
                    self.expert_attn_rank,
                    self.expert_attn_alpha,
                    **common,
                ),
                "ffn": LoRAConfig(
                    self.expert_ffn_rank,
                    self.expert_ffn_alpha,
                    **common,
                ),
            },
        }


class LoRALinear(nn.Linear):
    """An ``nn.Linear`` with a zero-initialized low-rank residual branch."""

    def __init__(self, *args, lora_config: LoRAConfig, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lora_config = lora_config
        self.lora_a = nn.Parameter(
            torch.empty(lora_config.rank, self.in_features, device=self.weight.device, dtype=self.weight.dtype)
        )
        self.lora_b = nn.Parameter(
            torch.zeros(self.out_features, lora_config.rank, device=self.weight.device, dtype=self.weight.dtype)
        )
        nn.init.normal_(self.lora_a, std=lora_config.init_std)
        self.lora_dropout = nn.Dropout(lora_config.dropout) if lora_config.dropout else nn.Identity()

    @classmethod
    def from_linear(cls, linear: nn.Linear, config: LoRAConfig) -> "LoRALinear":
        module = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
            lora_config=config,
        )
        with torch.no_grad():
            module.weight.copy_(linear.weight)
            if linear.bias is not None:
                module.bias.copy_(linear.bias)
        return module

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = F.linear(inputs, self.weight, self.bias)
        residual = F.linear(F.linear(self.lora_dropout(inputs), self.lora_a), self.lora_b)
        return base + residual * self.lora_config.scale

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        return self.weight + (self.lora_b @ self.lora_a) * self.lora_config.scale


_ATTN = {"q_proj", "k_proj", "v_proj", "o_proj"}
_FFN = {"gate_proj", "up_proj", "down_proj"}


def _target_backbone(
    full_name: str,
    leaf: str,
    config: LoRATrainingConfig,
) -> str | None:
    if leaf not in config.target_modules:
        return None
    if "vision_tower" in full_name or "vision_model" in full_name:
        return None
    paligemma = "paligemma" in full_name and "language_model" in full_name
    expert = "gemma_expert" in full_name
    if config.apply_to in {"paligemma_attn", "expert_attn"} and leaf not in _ATTN:
        return None
    if config.apply_to in {"all"}:
        if paligemma:
            return "paligemma"
        if expert:
            return "expert"
        return None
    if config.apply_to in {"paligemma_only", "paligemma_attn"}:
        return "paligemma" if paligemma else None
    return "expert" if expert else None


def inject_lora(model: nn.Module, config: LoRATrainingConfig) -> list[str]:
    """Replace selected Gemma projections and return their full module names."""
    if not config.enabled:
        return []
    layer_configs = config.configs()
    replaced: list[str] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or isinstance(module, LoRALinear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        backbone = _target_backbone(name, leaf, config)
        if backbone is None:
            continue
        kind = "attn" if leaf in _ATTN else "ffn" if leaf in _FFN else None
        if kind is None:
            continue
        parent_name, _, attribute = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(
            parent,
            attribute,
            LoRALinear.from_linear(module, layer_configs[backbone][kind]),
        )
        replaced.append(name)
    if not replaced:
        raise ValueError("LoRA did not match any linear layers; check apply_to/target_modules")
    return replaced


def configure_lora_training(model: nn.Module, config: LoRATrainingConfig) -> dict[str, int]:
    """Freeze the base model and enable adapters plus explicitly requested heads."""
    counts = {"lora": 0, "vision": 0, "non_lora": 0, "frozen": 0}
    for name, parameter in model.named_parameters():
        is_lora = name.endswith("lora_a") or name.endswith("lora_b")
        is_vision = "vision_tower" in name or "vision_model" in name
        is_extra = config.train_non_lora_layers and any(
            token in name for token in config.trainable_modules
        )
        train = is_lora or (is_vision and config.train_vision_encoder) or is_extra
        parameter.requires_grad_(train)
        role = "lora" if is_lora else "vision" if is_vision and train else "non_lora" if is_extra else "frozen"
        counts[role] += parameter.numel()
    if counts["lora"] == 0:
        raise ValueError("No trainable LoRA parameters found; call inject_lora first")
    return counts


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return adapters and any explicitly trainable non-LoRA tensors."""
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name in trainable
    }


def save_lora_adapter(model: nn.Module, directory: str | Path, config: LoRATrainingConfig) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file(adapter_state_dict(model), directory / "adapter_model.safetensors")
    (directory / "adapter_config.json").write_text(
        json.dumps(dataclasses.asdict(config), indent=2, sort_keys=True), encoding="utf-8"
    )


def load_lora_adapter(model: nn.Module, directory: str | Path, *, strict: bool = True) -> None:
    state = safetensors.torch.load_file(str(Path(directory) / "adapter_model.safetensors"))
    result = model.load_state_dict(state, strict=False)
    unexpected = list(result.unexpected_keys)
    missing_adapter = [
        name for name, _ in model.named_parameters()
        if (name.endswith("lora_a") or name.endswith("lora_b")) and name not in state
    ]
    if strict and (unexpected or missing_adapter):
        raise RuntimeError(
            f"Adapter mismatch: unexpected={unexpected}, missing={missing_adapter}"
        )


__all__ = [
    "LoRAConfig", "LoRALinear", "LoRATrainingConfig", "adapter_state_dict",
    "configure_lora_training", "inject_lora", "load_lora_adapter", "save_lora_adapter",
]

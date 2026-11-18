"""Static configuration objects for Pi0 / Pi0.5.

Self-contained: no jax / flax / openpi imports.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

PALIGEMMA_VOCAB_SIZE = 257_152 # PaliGemma vocabulary size (also used as the special image-token index).

GemmaVariant = Literal["dummy", "gemma_300m", "gemma_2b"]


@dataclasses.dataclass(frozen=True) # Keep fronzen so cannot be modified after creation.
class GemmaConfig:
    """Architecture sizes for one Gemma stack (used for both the PaliGemma LM and the action expert)."""

    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int


_GEMMA_CONFIGS: dict[str, GemmaConfig] = {
    "dummy": GemmaConfig(width=64, depth=4, mlp_dim=128, num_heads=8, num_kv_heads=1, head_dim=16),
    "gemma_300m": GemmaConfig(width=1024, depth=18, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256), # 311M params
    "gemma_2b": GemmaConfig(width=2048, depth=18, mlp_dim=16_384, num_heads=8, num_kv_heads=1, head_dim=256),
}


def get_gemma_config(variant: GemmaVariant) -> GemmaConfig:
    if variant not in _GEMMA_CONFIGS:
        raise ValueError(f"Unknown gemma variant: {variant!r}. Known: {sorted(_GEMMA_CONFIGS)}")
    return _GEMMA_CONFIGS[variant]


# ---------------------------------------------------------------------------
# Pi0 / Pi0.5 model config
# ---------------------------------------------------------------------------
_PYTORCH_COMPILE_MODES = {
    None,
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
}


@dataclasses.dataclass(frozen=True)
class Pi0Config:
    """End-to-end Pi0 / Pi0.5 model configuration.

    The two models share the same backbone (PaliGemma + an action expert Gemma stack) but differ in:
      * `pi05=True` uses AdaRMSNorm in the action expert to inject the
        flow-matching timestep, and feeds the proprioceptive state as a
        discrete language token instead of through a continuous projection.
      * Defaults: pi05 -> action_horizon=10, max_token_len=200
                  pi0  -> action_horizon=50, max_token_len=48
    """

    # Architecture variants.
    paligemma_variant: GemmaVariant = "gemma_2b"
    action_expert_variant: GemmaVariant = "gemma_300m"

    # Action / token shapes.
    action_dim: int = 32
    action_horizon: int | None = None
    max_token_len: int | None = None

    # Pi0 vs Pi0.5 toggle.
    pi05: bool = False

    # Numerical precision used inside `PaliGemmaWithExpertModel`.
    dtype: Literal["bfloat16", "float32"] = "bfloat16"

    # If non-None, `sample_actions` will be wrapped in `torch.compile(...)`.
    pytorch_compile_mode: str | None = None

    def __post_init__(self) -> None:
        if self.action_horizon is None:
            object.__setattr__(self, "action_horizon", 10 if self.pi05 else 50)
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.pytorch_compile_mode not in _PYTORCH_COMPILE_MODES:
            raise ValueError(
                f"pytorch_compile_mode={self.pytorch_compile_mode!r} not in {_PYTORCH_COMPILE_MODES}"
            )

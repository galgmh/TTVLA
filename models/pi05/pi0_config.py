"""π0.5 / π0 的 PyTorch 模型配置（不依赖 JAX）。

字段与官方 openpi ``pi0_config.Pi0Config`` 对齐；PyTorch 实现只读取本文件中的
字段，不再提供 JAX 侧的 ``create`` / ``inputs_spec`` / 冻结过滤器等接口。

原官方 ``models/gemma.py`` 的 Gemma 结构配置（``Config`` / ``Variant`` /
``get_config``）也并入本文件，PyTorch 仓库中只保留这一处入口。
"""

from __future__ import annotations

import dataclasses
from typing import Literal

from models.pi05.thermal.config import ThermalPi0Config

#: PaliGemma 词表大小。
PALIGEMMA_VOCAB_SIZE = 257_152


@dataclasses.dataclass
class Config:
    """Gemma 主干的结构超参数。"""

    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int


#: PyTorch 路径支持的变体；LoRA 变体已随 JAX 实现一并移除。
Variant = Literal["dummy", "gemma_300m", "gemma_2b"]


def get_config(variant: Variant) -> Config:
    """返回指定 Gemma 变体的结构配置。"""
    if variant == "dummy":
        return Config(
            width=64,
            depth=4,
            mlp_dim=128,
            num_heads=8,
            num_kv_heads=1,
            head_dim=16,
        )
    if variant == "gemma_300m":
        # 311M 参数。
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    raise ValueError(
        f"Unknown variant: {variant}；PyTorch 路径只支持 'dummy' / 'gemma_300m' / 'gemma_2b'"
        "（LoRA 变体已随 JAX 实现移除）"
    )


@dataclasses.dataclass(frozen=True)
class Pi0Config:
    """一次模型构建的静态配置。"""

    dtype: str = "bfloat16"
    paligemma_variant: Variant = "gemma_2b"
    action_expert_variant: Variant = "gemma_300m"

    # 动作维度与预测时域。
    action_dim: int = 32
    action_horizon: int = 50
    # 分词后的 prompt 最大长度；None 时按 pi05 自动取 200 / 48。
    max_token_len: int | None = None
    # π0.5 与 π0 的两个差异：
    # - state 作为离散语言 token 的一部分，而不是 suffix 的连续输入；
    # - action expert 用 adaRMSNorm 注入 flow matching 时间步。
    pi05: bool = False
    # 该选项不被模型直接使用，但官方数据变换会读取它。
    discrete_state_input: bool | None = None

    pytorch_compile_mode: str | None = "max-autotune"

    # Thermal-π0.5 热觉分支配置；None 或 enabled=False 时保持原 π0.5 行为。
    thermal: ThermalPi0Config | None = None

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]


__all__ = [
    "PALIGEMMA_VOCAB_SIZE",
    "Config",
    "Pi0Config",
    "Variant",
    "get_config",
]

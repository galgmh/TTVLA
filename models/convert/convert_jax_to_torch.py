#!/usr/bin/env python3
"""Convert a Pi0 / Pi0.5 JAX (Orbax) checkpoint into a PyTorch safetensors file.

Simplified from openpi/examples/convert_jax_model_to_pytorch.py:
  - No dependency on openpi.training.config (pass model hparams as CLI args).
  - Explicit --pi05 flag instead of string-matching the checkpoint path.
  - Single self-contained file.

Still reuses the following from the OriginalCodes/openpi package (added to
sys.path at runtime):
  - openpi.models.model.restore_params       (orbax -> numpy dict)
  - openpi.models.pi0_config.Pi0Config       (model config dataclass)
  - openpi.models.gemma.get_config           (action expert dims)
  - openpi.models_pytorch.pi0_pytorch.PI0Pytorch  (target module for state_dict)

Later steps in SimplerCodes will replace these with local minimal copies.

Example:
    python convert_jax_to_torch.py \\
        --checkpoint_dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_base \\
        --output_path   ~/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch \\
        --pi05
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import sys
from typing import Literal

# --- Make the (locally vendored) openpi subset importable -------------------
# The minimal set of openpi modules needed for conversion lives at
# ./original_dependencies/openpi/**. Adding the parent directory to sys.path
# lets `import openpi.xxx` resolve to that vendored copy.
# We also add this directory itself so `import transformers_patches` works.
_HERE = pathlib.Path(__file__).resolve().parent
_OPENPI_ROOT = _HERE / "original_dependencies"
for p in (_HERE, _OPENPI_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Inject the openpi-patched Gemma / PaliGemma / SigLIP modules into the
# transformers namespace via sys.modules. MUST run before any openpi PyTorch
# model code is imported, otherwise the LazyModule in `transformers` will have
# already cached the unpatched class objects.
import transformers_patches  # noqa: E402

transformers_patches.install()

import numpy as np  # noqa: E402
import safetensors.torch  # noqa: E402
import torch  # noqa: E402
import tyro  # noqa: E402
from flax.nnx import traversals  # noqa: E402

import openpi.models.gemma as _gemma  # noqa: E402
import openpi.models.model as _model  # noqa: E402
import openpi.models.pi0_config as _pi0_config  # noqa: E402
import openpi.models_pytorch.pi0_pytorch as _pi0_pytorch  # noqa: E402


# ---------------------------------------------------------------------------
# PaliGemma (vision tower + language model) weight slicing
# ---------------------------------------------------------------------------
def _slice_paligemma(state_dict: dict, vision_cfg, text_cfg) -> tuple[dict, dict]:
    """JAX PaliGemma params -> PyTorch state_dict keys.

    Returns (paligemma_state_dict, leftover_expert_params).
    """
    suffix = "/value" if "img/embedding/kernel/value" in state_dict else ""
    PG = "paligemma_with_expert.paligemma.model"

    # --- patch + pos embeddings ---------------------------------------------
    state_dict[f"{PG}.vision_tower.vision_model.embeddings.patch_embedding.weight"] = (
        state_dict.pop(f"img/embedding/kernel{suffix}").transpose(3, 2, 0, 1)
    )
    state_dict[f"{PG}.vision_tower.vision_model.embeddings.patch_embedding.bias"] = state_dict.pop(
        f"img/embedding/bias{suffix}"
    )
    state_dict[f"{PG}.vision_tower.vision_model.embeddings.position_embedding.weight"] = state_dict.pop(
        f"img/pos_embedding{suffix}"
    ).reshape(-1, vision_cfg.hidden_size)

    # --- vision encoder blocks (stacked along axis 0) -----------------------
    ln0_s = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_0/scale{suffix}")
    ln0_b = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_0/bias{suffix}")
    ln1_s = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_1/scale{suffix}")
    ln1_b = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_1/bias{suffix}")

    mlp0_k = state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel{suffix}")
    mlp0_b = state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_0/bias{suffix}")
    mlp1_k = state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel{suffix}")
    mlp1_b = state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_1/bias{suffix}")

    mha = "img/Transformer/encoderblock/MultiHeadDotProductAttention_0"
    k_k = state_dict.pop(f"{mha}/key/kernel{suffix}")
    k_b = state_dict.pop(f"{mha}/key/bias{suffix}")
    v_k = state_dict.pop(f"{mha}/value/kernel{suffix}")
    v_b = state_dict.pop(f"{mha}/value/bias{suffix}")
    q_k = state_dict.pop(f"{mha}/query/kernel{suffix}")
    q_b = state_dict.pop(f"{mha}/query/bias{suffix}")
    o_k = state_dict.pop(f"{mha}/out/kernel{suffix}")
    o_b = state_dict.pop(f"{mha}/out/bias{suffix}")

    H = vision_cfg.hidden_size
    for i in range(vision_cfg.num_hidden_layers):
        base = f"{PG}.vision_tower.vision_model.encoder.layers.{i}"
        state_dict[f"{base}.layer_norm1.weight"] = ln0_s[i].transpose()
        state_dict[f"{base}.layer_norm1.bias"] = ln0_b[i]
        state_dict[f"{base}.layer_norm2.weight"] = ln1_s[i].transpose()
        state_dict[f"{base}.layer_norm2.bias"] = ln1_b[i]
        state_dict[f"{base}.mlp.fc1.weight"] = mlp0_k[i].transpose()
        state_dict[f"{base}.mlp.fc1.bias"] = mlp0_b[i]
        state_dict[f"{base}.mlp.fc2.weight"] = mlp1_k[i].transpose()
        state_dict[f"{base}.mlp.fc2.bias"] = mlp1_b[i]
        state_dict[f"{base}.self_attn.k_proj.weight"] = k_k[i].reshape(-1, H).transpose()
        state_dict[f"{base}.self_attn.k_proj.bias"] = k_b[i].reshape(-1)
        state_dict[f"{base}.self_attn.v_proj.weight"] = v_k[i].reshape(-1, H).transpose()
        state_dict[f"{base}.self_attn.v_proj.bias"] = v_b[i].reshape(-1)
        state_dict[f"{base}.self_attn.q_proj.weight"] = q_k[i].reshape(-1, H).transpose()
        state_dict[f"{base}.self_attn.q_proj.bias"] = q_b[i].reshape(-1)
        state_dict[f"{base}.self_attn.out_proj.weight"] = o_k[i].reshape(-1, H).transpose()
        state_dict[f"{base}.self_attn.out_proj.bias"] = o_b[i].reshape(-1)

    # --- vision post-layernorm + multimodal projector -----------------------
    state_dict[f"{PG}.vision_tower.vision_model.post_layernorm.weight"] = state_dict.pop(
        f"img/Transformer/encoder_norm/scale{suffix}"
    ).transpose()
    state_dict[f"{PG}.vision_tower.vision_model.post_layernorm.bias"] = state_dict.pop(
        f"img/Transformer/encoder_norm/bias{suffix}"
    )
    state_dict[f"{PG}.multi_modal_projector.linear.weight"] = state_dict.pop(f"img/head/kernel{suffix}").transpose()
    state_dict[f"{PG}.multi_modal_projector.linear.bias"] = state_dict.pop(f"img/head/bias{suffix}")

    # --- language model (gemma) ---------------------------------------------
    state_dict[f"{PG}.language_model.embed_tokens.weight"] = state_dict.pop(
        f"llm/embedder/input_embedding{suffix}"
    )

    attn_out = state_dict.pop(f"llm/layers/attn/attn_vec_einsum/w{suffix}")
    attn_kv = state_dict.pop(f"llm/layers/attn/kv_einsum/w{suffix}")
    attn_q = state_dict.pop(f"llm/layers/attn/q_einsum/w{suffix}")
    mlp_gate = state_dict.pop(f"llm/layers/mlp/gating_einsum{suffix}")
    mlp_lin = state_dict.pop(f"llm/layers/mlp/linear{suffix}")
    pre_attn_ln = state_dict.pop(f"llm/layers/pre_attention_norm/scale{suffix}")
    pre_ffw_ln = state_dict.pop(f"llm/layers/pre_ffw_norm/scale{suffix}")

    n_h = text_cfg.num_attention_heads
    d_h = text_cfg.head_dim
    d_m = text_cfg.hidden_size
    for i in range(text_cfg.num_hidden_layers):
        base = f"{PG}.language_model.layers.{i}"
        state_dict[f"{base}.self_attn.q_proj.weight"] = attn_q[i].transpose(0, 2, 1).reshape(n_h * d_h, d_m)
        state_dict[f"{base}.self_attn.k_proj.weight"] = attn_kv[i, 0, 0].transpose()
        state_dict[f"{base}.self_attn.v_proj.weight"] = attn_kv[i, 1, 0].transpose()
        state_dict[f"{base}.self_attn.o_proj.weight"] = (
            attn_out[i].transpose(2, 0, 1).reshape(n_h * d_h, d_m)
        )
        state_dict[f"{base}.mlp.gate_proj.weight"] = mlp_gate[i, 0].transpose()
        state_dict[f"{base}.mlp.up_proj.weight"] = mlp_gate[i, 1].transpose()
        state_dict[f"{base}.mlp.down_proj.weight"] = mlp_lin[i].transpose()
        state_dict[f"{base}.input_layernorm.weight"] = pre_attn_ln[i]
        state_dict[f"{base}.post_attention_layernorm.weight"] = pre_ffw_ln[i]

    state_dict[f"{PG}.language_model.norm.weight"] = state_dict.pop(f"llm/final_norm/scale{suffix}")

    # Separate expert (action-expert) params from the rest.
    expert_keys = {
        f"llm/final_norm_1/scale{suffix}",
        f"llm/final_norm_1/Dense_0/bias{suffix}",
        f"llm/final_norm_1/Dense_0/kernel{suffix}",
        f"llm/layers/attn/attn_vec_einsum_1/w{suffix}",
        f"llm/layers/attn/kv_einsum_1/w{suffix}",
        f"llm/layers/attn/q_einsum_1/w{suffix}",
        f"llm/layers/mlp_1/gating_einsum{suffix}",
        f"llm/layers/mlp_1/linear{suffix}",
        f"llm/layers/pre_attention_norm_1/scale{suffix}",
        f"llm/layers/pre_attention_norm_1/Dense_0/bias{suffix}",
        f"llm/layers/pre_attention_norm_1/Dense_0/kernel{suffix}",
        f"llm/layers/pre_ffw_norm_1/scale{suffix}",
        f"llm/layers/pre_ffw_norm_1/Dense_0/bias{suffix}",
        f"llm/layers/pre_ffw_norm_1/Dense_0/kernel{suffix}",
    }
    paligemma_sd, expert_sd = {}, {}
    for k, v in state_dict.items():
        if k in expert_keys:
            expert_sd[k] = v
        else:
            paligemma_sd[k] = torch.from_numpy(v)
    return paligemma_sd, expert_sd


# ---------------------------------------------------------------------------
# Gemma action-expert weight slicing
# ---------------------------------------------------------------------------
def _slice_expert(state_dict: dict, cfg, *, num_expert: int, pi05: bool) -> dict:
    """JAX action-expert params -> PyTorch state_dict keys."""
    # Pad config with HF-style attribute names used below.
    if not hasattr(cfg, "hidden_size"):
        cfg.hidden_size = cfg.width
    if not hasattr(cfg, "num_hidden_layers"):
        cfg.num_hidden_layers = cfg.depth
    if not hasattr(cfg, "num_attention_heads"):
        cfg.num_attention_heads = cfg.num_heads

    suffix = "/value" if f"llm/layers/attn/attn_vec_einsum_{num_expert}/w/value" in state_dict else ""
    EX = "paligemma_with_expert.gemma_expert.model"

    attn_out = state_dict.pop(f"llm/layers/attn/attn_vec_einsum_{num_expert}/w{suffix}")
    attn_kv = state_dict.pop(f"llm/layers/attn/kv_einsum_{num_expert}/w{suffix}")
    attn_q = state_dict.pop(f"llm/layers/attn/q_einsum_{num_expert}/w{suffix}")
    mlp_gate = state_dict.pop(f"llm/layers/mlp_{num_expert}/gating_einsum{suffix}")
    mlp_lin = state_dict.pop(f"llm/layers/mlp_{num_expert}/linear{suffix}")

    if pi05:
        pre_attn_b = state_dict.pop(f"llm/layers/pre_attention_norm_{num_expert}/Dense_0/bias{suffix}")
        pre_ffw_b = state_dict.pop(f"llm/layers/pre_ffw_norm_{num_expert}/Dense_0/bias{suffix}")
        pre_attn_w = state_dict.pop(f"llm/layers/pre_attention_norm_{num_expert}/Dense_0/kernel{suffix}")
        pre_ffw_w = state_dict.pop(f"llm/layers/pre_ffw_norm_{num_expert}/Dense_0/kernel{suffix}")
    else:
        pre_attn_ln = state_dict.pop(f"llm/layers/pre_attention_norm_{num_expert}/scale{suffix}")
        pre_ffw_ln = state_dict.pop(f"llm/layers/pre_ffw_norm_{num_expert}/scale{suffix}")

    n_h, d_h, d_m = cfg.num_attention_heads, cfg.head_dim, cfg.hidden_size
    for i in range(cfg.num_hidden_layers):
        base = f"{EX}.layers.{i}"
        state_dict[f"{base}.self_attn.q_proj.weight"] = attn_q[i].transpose(0, 2, 1).reshape(n_h * d_h, d_m)
        state_dict[f"{base}.self_attn.k_proj.weight"] = attn_kv[i, 0, 0].transpose()
        state_dict[f"{base}.self_attn.v_proj.weight"] = attn_kv[i, 1, 0].transpose()
        state_dict[f"{base}.self_attn.o_proj.weight"] = attn_out[i].reshape(n_h * d_h, d_m).transpose(1, 0)
        state_dict[f"{base}.mlp.gate_proj.weight"] = mlp_gate[i, 0].transpose()
        state_dict[f"{base}.mlp.up_proj.weight"] = mlp_gate[i, 1].transpose()
        state_dict[f"{base}.mlp.down_proj.weight"] = mlp_lin[i].transpose()
        if pi05:
            state_dict[f"{base}.input_layernorm.dense.bias"] = pre_attn_b[i]
            state_dict[f"{base}.post_attention_layernorm.dense.bias"] = pre_ffw_b[i]
            state_dict[f"{base}.input_layernorm.dense.weight"] = pre_attn_w[i].transpose()
            state_dict[f"{base}.post_attention_layernorm.dense.weight"] = pre_ffw_w[i].transpose()
        else:
            state_dict[f"{base}.input_layernorm.weight"] = pre_attn_ln[i]
            state_dict[f"{base}.post_attention_layernorm.weight"] = pre_ffw_ln[i]

    if pi05:
        state_dict[f"{EX}.norm.dense.bias"] = state_dict.pop(f"llm/final_norm_{num_expert}/Dense_0/bias{suffix}")
        state_dict[f"{EX}.norm.dense.weight"] = state_dict.pop(
            f"llm/final_norm_{num_expert}/Dense_0/kernel{suffix}"
        ).transpose()
    else:
        state_dict[f"{EX}.norm.weight"] = state_dict.pop(f"llm/final_norm_{num_expert}/scale{suffix}")

    return {k: (v if isinstance(v, torch.Tensor) else torch.from_numpy(v)) for k, v in state_dict.items()}


# ---------------------------------------------------------------------------
# Projection heads (state/action/time MLPs sit outside PaliGemma+Expert)
# ---------------------------------------------------------------------------
def _slice_projections(projection_params: dict, *, pi05: bool) -> dict:
    keys = (
        ["action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"]
        if pi05
        else ["state_proj", "action_in_proj", "action_out_proj", "action_time_mlp_in", "action_time_mlp_out"]
    )
    out = {}
    for k in keys:
        kernel = projection_params[k]["kernel"]
        bias = projection_params[k]["bias"]
        if isinstance(kernel, dict):  # "value" suffix not stripped
            kernel, bias = kernel["value"], bias["value"]
        out[f"{k}.weight"] = torch.from_numpy(np.array(kernel)).T
        out[f"{k}.bias"] = torch.from_numpy(np.array(bias))
    return out


# ---------------------------------------------------------------------------
# Hard-coded PaliGemma architecture dims (matches openpi's conversion script).
# ---------------------------------------------------------------------------
class _PaliGemmaArch:
    class vision_config:
        hidden_size = 1152
        num_hidden_layers = 27
        num_attention_heads = 16
        intermediate_size = 4304
        patch_size = 14
        projection_dim = 2048

    class text_config:
        hidden_size = 2048
        num_hidden_layers = 18
        num_attention_heads = 8
        head_dim = 256
        intermediate_size = 16384


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------
def convert(
    checkpoint_dir: str,
    output_path: str,
    *,
    pi05: bool = False,
    action_dim: int = 32,
    action_horizon: int | None = None,
    max_token_len: int | None = None,
    paligemma_variant: str = "gemma_2b",
    action_expert_variant: str = "gemma_300m",
    precision: Literal["float32", "bfloat16"] = "bfloat16",
) -> None:
    """Load a JAX Orbax checkpoint and save a PyTorch safetensors checkpoint."""

    # Default hparams consistent with openpi's Pi0Config __post_init__.
    if action_horizon is None:
        action_horizon = 10 if pi05 else 50
    if max_token_len is None:
        max_token_len = 200 if pi05 else 48

    model_config = _pi0_config.Pi0Config(
        action_dim=action_dim,
        action_horizon=action_horizon,
        max_token_len=max_token_len,
        pi05=pi05,
        paligemma_variant=paligemma_variant,
        action_expert_variant=action_expert_variant,
    )
    print(f"[convert] model_config = {model_config}")

    # 1. Load raw JAX params as numpy (force float32 to match reference script).
    print(f"[convert] restoring params from {checkpoint_dir}/params ...")
    params = _model.restore_params(
        f"{checkpoint_dir}/params/", restore_type=np.ndarray, dtype="float32"
    )
    paligemma_flat = traversals.flatten_mapping(params["PaliGemma"], sep="/")

    # 2. Slice.
    paligemma_sd, expert_flat = _slice_paligemma(
        paligemma_flat, _PaliGemmaArch.vision_config, _PaliGemmaArch.text_config
    )
    expert_arch = _gemma.get_config(action_expert_variant)
    expert_sd = _slice_expert(expert_flat, expert_arch, num_expert=1, pi05=pi05)
    proj_sd = _slice_projections(params, pi05=pi05)

    all_params = {**paligemma_sd, **expert_sd, **proj_sd}

    # 3. Instantiate the PyTorch model and load.
    print("[convert] instantiating PI0Pytorch and loading weights ...")
    pi0 = _pi0_pytorch.PI0Pytorch(model_config)
    missing, unexpected = pi0.load_state_dict(all_params, strict=False)
    if unexpected:
        print(f"[convert][warn] {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")
    if missing:
        # A few buffers / tied weights may show up here - print for visibility.
        print(f"[convert][warn] {len(missing)} missing keys, e.g. {missing[:3]}")

    pi0 = pi0.to({"float32": torch.float32, "bfloat16": torch.bfloat16}[precision])

    # 4. Save safetensors (use save_model so tied weights are handled).
    os.makedirs(output_path, exist_ok=True)
    safetensors.torch.save_model(pi0, os.path.join(output_path, "model.safetensors"))

    # Copy sibling `assets/` if present (norm stats etc.).
    assets_src = pathlib.Path(checkpoint_dir).parent / "assets"
    if assets_src.exists():
        assets_dst = pathlib.Path(output_path) / "assets"
        if assets_dst.exists():
            shutil.rmtree(assets_dst)
        shutil.copytree(assets_src, assets_dst)

    # Save a minimal config for later loading.
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(
            {
                "pi05": pi05,
                "action_dim": action_dim,
                "action_horizon": action_horizon,
                "max_token_len": max_token_len,
                "paligemma_variant": paligemma_variant,
                "action_expert_variant": action_expert_variant,
                "precision": precision,
            },
            f,
            indent=2,
        )
    print(f"[convert] done -> {output_path}")


if __name__ == "__main__":
    tyro.cli(convert)

"""PaliGemma + Gemma-action-expert dual-stream wrapper.

Still depends on the openpi-patched HuggingFace `transformers` modules; load
those via `pi05_model._patches.install()` BEFORE importing this file.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from transformers import GemmaForCausalLM, PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma

from .config import GemmaConfig, PALIGEMMA_VOCAB_SIZE


class PaliGemmaWithExpertModel(nn.Module):
    """Two parallel Gemma stacks sharing attention computation per layer.

    Stack A: the PaliGemma vision-language model (vision tower + LM).
    Stack B: an action-expert Gemma stack (LM only, embed_tokens disabled).
    Each forward pass interleaves the two streams at every transformer block.
    """

    def __init__(
        self,
        vlm_config: GemmaConfig,
        action_expert_config: GemmaConfig,
        use_adarms: list[bool] | None = None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ) -> None:
        super().__init__()
        if use_adarms is None:
            use_adarms = [False, False]

        # PaliGemma (vision + LM) HF config.
        vlm_hf = CONFIG_MAPPING["paligemma"]()
        vlm_hf._vocab_size = PALIGEMMA_VOCAB_SIZE  # noqa: SLF001
        vlm_hf.image_token_index = PALIGEMMA_VOCAB_SIZE
        vlm_hf.text_config.hidden_size = vlm_config.width
        vlm_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_hf.text_config.head_dim = vlm_config.head_dim
        vlm_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_hf.text_config.torch_dtype = "float32"
        vlm_hf.text_config.vocab_size = PALIGEMMA_VOCAB_SIZE
        vlm_hf.text_config.use_adarms = use_adarms[0]
        vlm_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_hf.vision_config.intermediate_size = 4304
        vlm_hf.vision_config.projection_dim = 2048
        vlm_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_hf.vision_config.torch_dtype = "float32"

        # Action-expert Gemma HF config.
        expert_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=PALIGEMMA_VOCAB_SIZE,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_hf)
        self.gemma_expert = GemmaForCausalLM(config=expert_hf)
        # The action expert never embeds language tokens itself, setting the embedding layer to None.
        self.gemma_expert.model.embed_tokens = None

        self._cast_precision(precision)

    # -----------------------------------------------------------------------
    # Precision casting
    # -----------------------------------------------------------------------
    def _cast_precision(self, precision: Literal["bfloat16", "float32"]) -> None:
        if precision == "float32":
            self.to(dtype=torch.float32)
            return
        if precision != "bfloat16":
            raise ValueError(f"Invalid precision: {precision!r}")

        self.to(dtype=torch.bfloat16)

        # A handful of params stay in float32 — they're either shape-sensitive
        # (vision patch embed) or normalization scales/biases that are easier
        # to train numerically in fp32.
        keep_fp32_substrings = (
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        )
        for name, param in self.named_parameters():
            if any(s in name for s in keep_fp32_substrings):
                param.data = param.data.to(dtype=torch.float32)

    # -----------------------------------------------------------------------
    # Cheap accessors used by PI0Pytorch.
    # -----------------------------------------------------------------------
    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.paligemma.language_model.embed_tokens(tokens)

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------
    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.Tensor | None] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor | None] | None = None,
    ):
        """Run one or both streams.

        `inputs_embeds` is a 2-list `[prefix_embs, suffix_embs]` where either entry may be None. Three regimes:

        * suffix is None  -> only the PaliGemma LM runs (used to build the KV cache from the image+language prefix).
        * prefix is None  -> only the action expert runs (used to apply a single denoising step against an existing prefix KV cache).
        * both present     -> per-layer interleaved attention across both stacks (used during training).
        """
        if adarms_cond is None:
            adarms_cond = [None, None]
        assert inputs_embeds is not None and len(inputs_embeds) == 2

        prefix_embs, suffix_embs = inputs_embeds

        # --- Single-stream paths -------------------------------------------
        if suffix_embs is None:
            out = self.paligemma.language_model.forward(
                inputs_embeds=prefix_embs,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0],
            )
            return [out.last_hidden_state, None], out.past_key_values

        if prefix_embs is None:
            out = self.gemma_expert.model.forward(
                inputs_embeds=suffix_embs,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1],
            )
            return [None, out.last_hidden_state], None

        # --- Dual-stream interleaved path ----------------------------------
        return self._forward_dual(
            prefix_embs=prefix_embs,
            suffix_embs=suffix_embs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            adarms_cond=adarms_cond,
        )

    def _forward_dual(
        self,
        *,
        prefix_embs: torch.Tensor,
        suffix_embs: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        adarms_cond: list[torch.Tensor | None],
    ):
        num_layers = self.paligemma.config.text_config.num_hidden_layers

        use_gc = (self.training and getattr(self.gemma_expert.model, "gradient_checkpointing", False))

        embeds = [prefix_embs, suffix_embs]
        for layer_idx in range(num_layers):
            if use_gc:
                embeds = torch.utils.checkpoint.checkpoint(
                    self._compute_layer,
                    layer_idx,
                    embeds,
                    attention_mask,
                    position_ids,
                    adarms_cond,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                embeds = self._compute_layer(layer_idx, embeds, attention_mask, position_ids, adarms_cond)

        # Final normalization per stream.
        if use_gc:
            outputs = torch.utils.checkpoint.checkpoint(
                self._compute_final_norms,
                embeds,
                adarms_cond,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            outputs = self._compute_final_norms(embeds, adarms_cond)

        return [outputs[0], outputs[1]], None

    # -----------------------------------------------------------------------
    # Per-layer dual-stream attention.
    # -----------------------------------------------------------------------
    def _compute_layer(
        self,
        layer_idx: int,
        inputs_embeds: list[torch.Tensor],
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        adarms_cond: list[torch.Tensor | None],
    ) -> list[torch.Tensor]:
        models = [self.paligemma.language_model, self.gemma_expert.model]

        query_states, key_states, value_states, gates = [], [], [], []
        for i, hidden_states in enumerate(inputs_embeds):
            layer = models[i].layers[layer_idx]
            hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])
            gates.append(gate)

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
            q = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            k = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            v = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            query_states.append(q)
            key_states.append(k)
            value_states.append(v)

        # Concatenate the two streams along the sequence dim and run a single joint attention.
        q_cat = torch.cat(query_states, dim=2)
        k_cat = torch.cat(key_states, dim=2)
        v_cat = torch.cat(value_states, dim=2)

        # Rotary embeddings come from the PaliGemma LM (shared).
        dummy = torch.zeros(
            q_cat.shape[0], q_cat.shape[2], q_cat.shape[-1],
            device=q_cat.device, dtype=q_cat.dtype,
        )
        cos, sin = self.paligemma.model.language_model.rotary_emb(dummy, position_ids)
        q_cat, k_cat = modeling_gemma.apply_rotary_pos_emb(q_cat, k_cat, cos, sin, unsqueeze_dim=1)

        scaling = self.paligemma.language_model.layers[layer_idx].self_attn.scaling
        att_output, _ = modeling_gemma.eager_attention_forward(
            self.paligemma.language_model.layers[layer_idx].self_attn,
            q_cat, k_cat, v_cat,
            attention_mask, scaling,
        )
        head_dim = self.paligemma.language_model.layers[layer_idx].self_attn.head_dim
        att_output = att_output.reshape(q_cat.shape[0], -1, 8 * head_dim)

        # Split attention output back across the two streams and run their
        # independent post-attention projections + MLPs.
        outputs_embeds = []
        start = 0
        for i, hidden_states in enumerate(inputs_embeds):
            layer = models[i].layers[layer_idx]
            end = start + hidden_states.shape[1]

            o_dtype = layer.self_attn.o_proj.weight.dtype
            slice_out = att_output[:, start:end]
            if slice_out.dtype != o_dtype:
                slice_out = slice_out.to(o_dtype)
            out_emb = layer.self_attn.o_proj(slice_out)

            # First (attention) residual.
            out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])  # noqa: SLF001
            after_first = out_emb.clone()

            out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
            if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                out_emb = out_emb.to(dtype=torch.bfloat16)
            out_emb = layer.mlp(out_emb)

            # Second (MLP) residual.
            out_emb = modeling_gemma._gated_residual(after_first, out_emb, gate)  # noqa: SLF001
            outputs_embeds.append(out_emb)
            start = end

        return outputs_embeds

    def _compute_final_norms(
        self,
        inputs_embeds: list[torch.Tensor],
        adarms_cond: list[torch.Tensor | None],
    ) -> list[torch.Tensor]:
        models = [self.paligemma.language_model, self.gemma_expert.model]
        outputs = []
        for i, hidden_states in enumerate(inputs_embeds):
            out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
            outputs.append(out_emb)
        return outputs

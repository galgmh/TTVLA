"""Pi0 / Pi0.5 PyTorch model: flow-matching policy on PaliGemma + action expert.

Lightly cleaned port of `openpi.models_pytorch.pi0_pytorch.PI0Pytorch`:
 * Local imports only (`.config`, `.gemma`, `.preprocessing`).
 * Removed the runtime sentinel that re-validated the `transformers_replace`
   monkey-patch — `pi05_model._patches.install()` covers that contract.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from .config import Pi0Config, get_gemma_config
from .gemma import PaliGemmaWithExpertModel
from .preprocessing import preprocess_observation_pytorch

logger = logging.getLogger(__name__)

# Large-negative constant used to mask out attention positions for fp32 / bf16.
_MASK_NEG = -2.3819763e38


def _safe_dtype(target: torch.dtype, device_type: str) -> torch.dtype:
    """CPU lacks bf16 kernels in many ops — fall back to fp32."""
    if device_type == "cpu" and target == torch.bfloat16:
        return torch.float32
    return target


def _sinusoidal_pos_embedding(
    time: torch.Tensor, dimension: int, *, min_period: float, max_period: float, device: torch.device
) -> Tensor:
    """Sin/cos positional embedding for scalar timesteps (B,) -> (B, dim)."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("time tensor must be 1-D (batch_size,)")

    dtype = _safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling = 1.0 / period * 2 * math.pi
    sin_input = scaling[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def _sample_beta(alpha: float, beta: float, bsize: int, device: torch.device) -> torch.Tensor:
    a = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    b = torch.as_tensor(beta, dtype=torch.float32, device=device)
    return torch.distributions.Beta(a, b).sample((bsize,))


def make_att_2d_masks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    """Build a 2D attention mask from pad + per-token AR-block markers.

    `att_masks` is an int[B, N] mask: each `1` opens a new attention block
    boundary, each `0` keeps the same block as the previous token. This lets
    us encode causal / prefix-lm / block-causal patterns with a single int
    array. See big_vision for the original.
    """
    if att_masks.ndim != 2 or pad_masks.ndim != 2:
        raise ValueError(f"expected 2D masks, got att={att_masks.ndim}D pad={pad_masks.ndim}D")

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d & pad_2d


class PI0Pytorch(nn.Module):
    """Flow-matching action policy: PaliGemma prefix + Gemma action expert."""

    def __init__(self, config: Pi0Config) -> None:
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_cfg = get_gemma_config(config.paligemma_variant)
        action_expert_cfg = get_gemma_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_cfg,
            action_expert_cfg,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(config.action_dim, action_expert_cfg.width)
        self.action_out_proj = nn.Linear(action_expert_cfg.width, config.action_dim)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_cfg.width, action_expert_cfg.width)
            self.time_mlp_out = nn.Linear(action_expert_cfg.width, action_expert_cfg.width)
        else:
            self.state_proj = nn.Linear(config.action_dim, action_expert_cfg.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_cfg.width, action_expert_cfg.width)
            self.action_time_mlp_out = nn.Linear(action_expert_cfg.width, action_expert_cfg.width)

        torch.set_float32_matmul_precision("high")
        if config.pytorch_compile_mode is not None:
            self.sample_actions = torch.compile(self.sample_actions, mode=config.pytorch_compile_mode)

        self.gradient_checkpointing_enabled = False

    # -----------------------------------------------------------------------
    # Gradient checkpointing on/off
    # -----------------------------------------------------------------------
    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        logger.info("PI0Pytorch: gradient checkpointing enabled")

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        logger.info("PI0Pytorch: gradient checkpointing disabled")

    def is_gradient_checkpointing_enabled(self) -> bool:
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------
    @staticmethod
    def _mask_to_4d_additive(att_2d_masks: torch.Tensor) -> torch.Tensor:
        """Convert (B, N, N) bool mask to (B, 1, N, N) additive float mask."""
        return torch.where(att_2d_masks[:, None, :, :], 0.0, _MASK_NEG)

    def _preprocess_observation(self, observation, *, train: bool):
        observation = preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device: torch.device) -> torch.Tensor:
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    def sample_time(self, bsize: int, device: torch.device) -> torch.Tensor:
        # Beta(1.5, 1.0) shifted away from {0,1} per the openpi recipe.
        time_beta = _sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    # -----------------------------------------------------------------------
    # Prefix (image + language) embedding
    # -----------------------------------------------------------------------
    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embs, pad_masks, att_marks = [], [], []

        for img, img_mask in zip(images, img_masks, strict=True):
            img_emb = self._apply_checkpoint(self.paligemma_with_expert.embed_image, img)
            bsize, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            # All image tokens share one attention block.
            att_marks += [0] * num_img_embs

        def lang_embed_func(t):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(t)
            return lang_emb * math.sqrt(lang_emb.shape[-1])

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        # Language tokens share the same attention block as images (full attention).
        att_marks += [0] * lang_emb.shape[1]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_marks_t = torch.tensor(att_marks, dtype=torch.bool, device=pad_masks.device)
        att_marks_t = att_marks_t[None, :].expand(pad_masks.shape[0], len(att_marks))
        return embs, pad_masks, att_marks_t

    # -----------------------------------------------------------------------
    # Suffix (state + noisy actions + timestep) embedding
    # -----------------------------------------------------------------------
    def embed_suffix(self, state, noisy_actions, timestep):
        embs, pad_masks, att_marks = [], [], []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)
            state_emb = self._apply_checkpoint(self.state_proj, state)
            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            pad_masks.append(torch.ones(bsize, 1, dtype=torch.bool, device=state_emb.device))
            # State opens a new attention block (cannot be attended by prefix).
            att_marks += [1]

        time_emb = _sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=4e-3,
            max_period=4.0,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        action_emb = self._apply_checkpoint(self.action_in_proj, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            def mlp_func(x):
                x = self.action_time_mlp_in(x)
                x = F.silu(x)
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            def time_mlp_func(x):
                x = self.time_mlp_in(x)
                x = F.silu(x)
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        pad_masks.append(torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device))

        # Open a new attention block at the first action token, then full
        # attention within the action block.
        att_marks += [1] + [0] * (self.config.action_horizon - 1)

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_marks_t = torch.tensor(att_marks, dtype=embs.dtype, device=embs.device)
        att_marks_t = att_marks_t[None, :].expand(bsize, len(att_marks))
        return embs, pad_masks, att_marks_t, adarms_cond

    # -----------------------------------------------------------------------
    # Training forward
    # -----------------------------------------------------------------------
    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Flow-matching training step. Returns per-element MSE loss tensor."""
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad, prefix_att = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad, suffix_att, adarms_cond = self.embed_suffix(state, x_t, time)

        # Cast embeddings to match LM weight dtype.
        if self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad, suffix_pad], dim=1)
        att_marks = torch.cat([prefix_att, suffix_att], dim=1)
        att_2d = make_att_2d_masks(pad_masks, att_marks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_4d = self._mask_to_4d_additive(att_2d)

        def forward_func(prefix_embs, suffix_embs, att_2d_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(forward_func, prefix_embs, suffix_embs, att_2d_4d, position_ids, adarms_cond)
        suffix_out = suffix_out[:, -self.config.action_horizon :].to(dtype=torch.float32)
        v_t = self._apply_checkpoint(self.action_out_proj, suffix_out)
        return F.mse_loss(u_t, v_t, reduction="none")

    # -----------------------------------------------------------------------
    # Inference: flow-matching denoising
    # -----------------------------------------------------------------------
    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps: int = 10) -> Tensor:
        """Flow-matching denoising inference. Returns (B, action_horizon, action_dim)."""
        bsize = observation.state.shape[0]
        if noise is None:
            noise = self.sample_noise((bsize, self.config.action_horizon, self.config.action_dim), device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad, prefix_att = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d = make_att_2d_masks(prefix_pad, prefix_att)
        prefix_position_ids = torch.cumsum(prefix_pad, dim=1) - 1
        prefix_att_2d_4d = self._mask_to_4d_additive(prefix_att_2d)

        # Building the prefix KV cache only needs the LM stack — force eager
        # attention so the cache layout matches what `denoise_step` expects.
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            v_t = self.denoise_step(state, prefix_pad, past_key_values, x_t, time.expand(bsize))
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        """One Euler step of the flow-matching ODE."""
        suffix_embs, suffix_pad, suffix_att, adarms_cond = self.embed_suffix(state, x_t, timestep)
        suffix_len = suffix_pad.shape[1]
        bsize = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
        suffix_att_2d = make_att_2d_masks(suffix_pad, suffix_att)
        full_att_2d = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad, dim=1) - 1

        full_att_2d_4d = self._mask_to_4d_additive(full_att_2d)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = outputs_embeds[1][:, -self.config.action_horizon :].to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)

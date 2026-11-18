"""Minimal Observation container consumed by `PI0Pytorch`.

Self-contained: replaces the heavyweight `openpi.models.model.Observation`
that comes with optional fields, jaxtyping checks, and frame conversions.
The Pi0/Pi0.5 PyTorch path only needs a small attribute bag.
"""

from __future__ import annotations

import dataclasses
from typing import Mapping

import torch


@dataclasses.dataclass
class Observation:
    """One batched observation step.

    Attributes
    ----------
    images:
        Mapping from camera key (e.g. "base_0_rgb", "left_wrist_0_rgb",
        "right_wrist_0_rgb") to an image tensor. Either NHWC or NCHW is
        accepted; values must be in [-1, 1] (float32/bfloat16) or [0, 255]
        (uint8). `preprocess_observation_pytorch` will resize / pad to 224x224.
    image_masks:
        Mapping from camera key to a bool tensor of shape (B,). True means the
        image is valid for that batch element. Optional — keys missing from
        this dict default to all-True.
    state:
        Proprioceptive state, shape (B, action_dim). For pi05 this is unused
        in the continuous projection path (the state is encoded as discrete
        language tokens by upstream tokenization).
    tokenized_prompt:
        Int tensor (B, T) of language token ids.
    tokenized_prompt_mask:
        Bool tensor (B, T) — True for real tokens, False for padding.
    token_ar_mask:
        Optional int tensor (B, T) used during training to mark autoregressive
        token boundaries. Inference-only paths can leave this as None.
    token_loss_mask:
        Optional bool tensor (B, T) used during training to mask the loss.
    """

    images: Mapping[str, torch.Tensor]
    image_masks: Mapping[str, torch.Tensor]
    state: torch.Tensor
    tokenized_prompt: torch.Tensor
    tokenized_prompt_mask: torch.Tensor
    token_ar_mask: torch.Tensor | None = None
    token_loss_mask: torch.Tensor | None = None

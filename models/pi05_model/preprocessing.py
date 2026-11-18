from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812

logger = logging.getLogger(__name__)

IMAGE_KEYS: tuple[str, ...] = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
IMAGE_RESOLUTION: tuple[int, int] = (224, 224)


# ---------------------------------------------------------------------------
# Aspect-ratio-preserving resize with zero/black padding (PyTorch).
# ---------------------------------------------------------------------------
def resize_with_pad_torch(
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Resize an image batch to (height, width) without distortion.

    Aspect ratio is preserved; the unused area is padded with 0 (uint8) or
    -1.0 (float32) so float images stay in [-1, 1].

    Accepts both NHWC and NCHW. Output keeps the input layout.
    """
    # Detect NHWC vs NCHW. If last-dim looks like a channel count, treat NHWC.
    channels_last = images.shape[-1] <= 4
    if channels_last:
        if images.dim() == 3:
            images = images.unsqueeze(0)
        images = images.permute(0, 3, 1, 2)
    elif images.dim() == 3:
        images = images.unsqueeze(0)

    cur_h, cur_w = images.shape[-2:]
    ratio = max(cur_w / width, cur_h / height)
    rh = int(cur_h / ratio)
    rw = int(cur_w / ratio)

    align_corners: bool | None = False if mode == "bilinear" else None
    resized = F.interpolate(images, size=(rh, rw), mode=mode, align_corners=align_corners)

    if images.dtype == torch.uint8:
        resized = torch.round(resized).clamp(0, 255).to(torch.uint8)
        pad_value: float = 0.0
    elif images.dtype in (torch.float32, torch.float16, torch.bfloat16):
        resized = resized.clamp(-1.0, 1.0)
        pad_value = -1.0
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    pad_h0, rem_h = divmod(height - rh, 2)
    pad_w0, rem_w = divmod(width - rw, 2)
    padded = F.pad(
        resized,
        (pad_w0, pad_w0 + rem_w, pad_h0, pad_h0 + rem_h),  # left, right, top, bottom
        mode="constant",
        value=pad_value,
    )

    if channels_last:
        padded = padded.permute(0, 2, 3, 1)

    return padded


# ---------------------------------------------------------------------------
# Observation preprocessing (resize + optional training augmentations).
# ---------------------------------------------------------------------------
class _ProcessedObservation:
    """Tiny attribute bag mirroring the fields the model reads."""

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


def preprocess_observation_pytorch(
    observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
):
    """Resize images to `image_resolution`, optionally augment, return a bag.

    The augmentation block (random crop / rotate / brightness / contrast /
    saturation) only fires when `train=True`. It deliberately uses
    tensor-only ops (no `.item()`) so it stays `torch.compile`-friendly.
    """
    if not set(image_keys).issubset(observation.images):
        missing = set(image_keys) - set(observation.images)
        raise ValueError(f"images dict missing keys {sorted(missing)}; got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]

    out_images: dict[str, torch.Tensor] = {}
    for key in image_keys:
        image = observation.images[key]

        # Accept either (B, C, H, W) or (B, H, W, C).
        is_channels_first = image.shape[1] == 3
        if is_channels_first:
            image = image.permute(0, 2, 3, 1)

        if image.shape[1:3] != tuple(image_resolution):
            logger.info(f"Resizing image {key} from {tuple(image.shape[1:3])} to {tuple(image_resolution)}")
            image = resize_with_pad_torch(image, *image_resolution)

        if train:
            image = _augment_train(image, key=key)

        if is_channels_first:
            image = image.permute(0, 3, 1, 2)

        out_images[key] = image

    out_masks: dict[str, torch.Tensor] = {}
    for key in out_images:
        if key in observation.image_masks:
            out_masks[key] = observation.image_masks[key]
        else:
            out_masks[key] = torch.ones(batch_shape, dtype=torch.bool, device=observation.state.device)

    return _ProcessedObservation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=getattr(observation, "token_ar_mask", None),
        token_loss_mask=getattr(observation, "token_loss_mask", None),
    )


# ---------------------------------------------------------------------------
# Training-time image augmentations (NHWC, values in [-1, 1] on entry/exit).
# ---------------------------------------------------------------------------
def _augment_train(image: torch.Tensor, *, key: str) -> torch.Tensor:
    # [-1, 1] -> [0, 1]
    image = image / 2.0 + 0.5

    # Geometric augmentations on non-wrist cameras.
    if "wrist" not in key:
        image = _random_crop_resize(image, scale=0.95)
        image = _random_small_rotation(image, max_deg=5.0)

    # Color augmentations.
    image = _random_brightness(image)
    image = _random_contrast(image)
    image = _random_saturation(image)

    image = torch.clamp(image, 0.0, 1.0)
    # [0, 1] -> [-1, 1]
    return image * 2.0 - 1.0


def _random_crop_resize(image: torch.Tensor, *, scale: float) -> torch.Tensor:
    """Random crop covering `scale * size`, then resize back."""
    h, w = image.shape[1:3]
    ch, cw = int(h * scale), int(w * scale)
    max_h, max_w = h - ch, w - cw
    if max_h <= 0 or max_w <= 0:
        return image
    start_h = torch.randint(0, max_h + 1, (1,), device=image.device)
    start_w = torch.randint(0, max_w + 1, (1,), device=image.device)
    cropped = image[:, start_h : start_h + ch, start_w : start_w + cw, :]
    resized = F.interpolate(
        cropped.permute(0, 3, 1, 2), size=(h, w), mode="bilinear", align_corners=False
    )
    return resized.permute(0, 2, 3, 1)


def _random_small_rotation(image: torch.Tensor, *, max_deg: float) -> torch.Tensor:
    """Apply a small random rotation via grid_sample."""
    angle = torch.rand(1, device=image.device) * (2 * max_deg) - max_deg
    if torch.abs(angle) <= 0.1:
        return image
    angle_rad = angle * math.pi / 180.0
    cos_a = torch.cos(angle_rad)
    sin_a = torch.sin(angle_rad)

    h, w = image.shape[1:3]
    grid_x = torch.linspace(-1, 1, w, device=image.device)
    grid_y = torch.linspace(-1, 1, h, device=image.device)
    grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
    grid_x = grid_x.unsqueeze(0).expand(image.shape[0], -1, -1)
    grid_y = grid_y.unsqueeze(0).expand(image.shape[0], -1, -1)
    grid = torch.stack(
        [grid_x * cos_a - grid_y * sin_a, grid_x * sin_a + grid_y * cos_a],
        dim=-1,
    )
    rotated = F.grid_sample(
        image.permute(0, 3, 1, 2), grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    return rotated.permute(0, 2, 3, 1)


def _random_brightness(image: torch.Tensor) -> torch.Tensor:
    factor = 0.7 + torch.rand(1, device=image.device) * 0.6  # [0.7, 1.3]
    return image * factor


def _random_contrast(image: torch.Tensor) -> torch.Tensor:
    factor = 0.6 + torch.rand(1, device=image.device) * 0.8  # [0.6, 1.4]
    mean = image.mean(dim=[1, 2, 3], keepdim=True)
    return (image - mean) * factor + mean


def _random_saturation(image: torch.Tensor) -> torch.Tensor:
    factor = 0.5 + torch.rand(1, device=image.device) * 1.0  # [0.5, 1.5]
    gray = image.mean(dim=-1, keepdim=True)
    return gray + (image - gray) * factor

"""Fine-tuning configurations for the current lift dataset."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from data.data_loader import DataConfig, RepackTransform
from finetune_new import TrainConfig
from pi05_model.config import Pi0Config
from pi05_model.lora import LoRATrainingConfig

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclasses.dataclass(frozen=True)
class ZeroRightWrist:
    """Single-arm robot: pad the unused right wrist view and mask it out."""

    def __call__(self, data: dict) -> dict:
        base = data["images"]["base_0_rgb"]
        return {
            **data,
            "images": {
                **data["images"],
                "right_wrist_0_rgb": np.zeros_like(base),
            },
            "image_masks": {
                "base_0_rgb": np.array(True),
                "left_wrist_0_rgb": np.array(True),
                "right_wrist_0_rgb": np.array(False),
            },
        }


DATA_CONFIG = DataConfig(
    repo_id="thermtacvla/lift",
    local_root=str("/root/autodl-fs/datasets/lerobot/lift"),
    asset_id="lift_norm_stats",
    norm_stats_dir=str(REPO_ROOT / "assets/lift_norm_stats"),
    use_quantile_norm=True,
    video_backend="pyav",
    repack_transforms=(
        RepackTransform(
            {
                "images": {
                    "base_0_rgb": "observation.images.up",
                    "left_wrist_0_rgb": "observation.images.side",
                },
                "state": "observation.state",
                "actions": "action",
                "task_index": "task_index",
            }
        ),
    ),
    data_transforms=(ZeroRightWrist(),),
    action_sequence_keys=("action",),
    prompt_from_task=True,
    default_prompt=None,
    tokenizer_path=str(
        "/root/autodl-tmp/openpi-cache/big_vision/paligemma_tokenizer.model"
    ),
)

MODEL_CONFIG = Pi0Config(pi05=True)

FULL_FINETUNE = TrainConfig(
    project_name="pi",
    exp_name="lift_ft",
    run_dir=str("/root/autodl-fs/runs"),
    log_interval=10,
    save_interval=4_000,
    resume=False,
    load_checkpoint_dir=None,
    wandb_enabled=True,
    model_config=None,
    pytorch_weight_path=str(
        "/root/autodl-fs/openpi/openpi-assets/checkpoints/pi05_base_pytorch"
    ),
    freeze_stage="vlm",
    pytorch_training_precision="bfloat16",
    warmup_steps=1,
    peak_lr=2.5e-5,
    decay_steps=15_000,
    end_lr=2.5e-6,
    optimizer_b1=0.9,
    optimizer_b2=0.95,
    optimizer_eps=1e-8,
    optimizer_weight_decay=1e-10,
    optimizer_clip_gradient_norm=1.0,
    num_train_steps=15_000,
    seed=42,
    batch_size=32,
    num_workers=4,
    zero_grad_double_check=True,
    data_config=None,
)

LORA_FINETUNE = dataclasses.replace(
    FULL_FINETUNE,
    exp_name="lift_lora",
    freeze_stage=None,
    peak_lr=1.4e-4,
    end_lr=1.4e-5,
    batch_size=16,
    num_train_steps=15_000,
    lora=LoRATrainingConfig(
        enabled=True,
        paligemma_attn_rank=16,
        paligemma_ffn_rank=16,
        paligemma_attn_alpha=16.0,
        paligemma_ffn_alpha=16.0,
        expert_attn_rank=32,
        expert_ffn_rank=32,
        expert_attn_alpha=32.0,
        expert_ffn_alpha=32.0,
        dropout=0.0,
        apply_to="all",
        train_vision_encoder=False,
        train_non_lora_layers=True,
    ),
)

ALLCONFIGS = {
    "lift_ft": {
        "data_config": DATA_CONFIG,
        "model_config": MODEL_CONFIG,
        "train_config": FULL_FINETUNE,
    },
    "lift_lora": {
        "data_config": DATA_CONFIG,
        "model_config": MODEL_CONFIG,
        "train_config": LORA_FINETUNE,
    },
}

__all__ = ["ALLCONFIGS"]

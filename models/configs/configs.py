from data.data_loader import DataConfig
from pi05_model.config import Pi0Config
from pi05_model.lora import LoRATrainingConfig
from finetune_new import TrainConfig
from data.data_loader import RepackTransform

import dataclasses
import pathlib
import numpy as np

# Repo root (models/ is one level down from it); used to keep cloud paths portable.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

@dataclasses.dataclass(frozen=True)
class ZeroRightWrist:
    """Single-arm robot: pad the unused right wrist view with zeros and mask it out."""

    def __call__(self, data: dict) -> dict:
        base = data["images"]["base_0_rgb"]      # repack 之后的输出
        return {
            **data,
            "images": {**data["images"], "right_wrist_0_rgb": np.zeros_like(base)},
            "image_masks": {                     # ← 新增的顶层字段
                "base_0_rgb": np.array(True),
                "left_wrist_0_rgb": np.array(True),
                "right_wrist_0_rgb": np.array(False),
            },
        }

ALLCONFIGS = {
    "test_run":{
        "data_config": DataConfig(
            repo_id = None,
            local_root = "/media/sunhao/T71/svla_so101_pickplace",
            asset_id = "svla_so101_pickplace_norm_stats+action_horizon_10",
            norm_stats_dir = "/media/sunhao/T71/Assets/svla_so101_pickplace_norm_stats+action_horizon_10",
            use_quantile_norm = True,
            repack_transforms=(
                RepackTransform({
                    "images": {
                        "base_0_rgb":       "observation.images.up",
                        "left_wrist_0_rgb": "observation.images.side",
                        # Note: LeRobot's dataset only has one wrist camera, so we use the same image for both left and right wrists.
                    },
                    "state":      "observation.state",
                    "actions":    "action",
                    "task_index": "task_index",
                }),
            ),
            data_transforms=(ZeroRightWrist(),),
            action_sequence_keys=("action",),         # LeRobot uses "action" here, not "actions"
            prompt_from_task=True,                   # raw sample already carries `prompt`
            default_prompt=None,
            tokenizer_path="/media/sunhao/T71/Pi0.5/paligemma_tokenizer/paligemma_tokenizer.model",
        ),
        "model_config": Pi0Config(pi05=True),
        "train_config": TrainConfig(
            project_name="pi",
            exp_name="test_run",
            run_dir="./runs",
            log_interval=10, # in steps
            save_interval=1000, # in steps
            resume=False, # If true, will resume training from load_checkpoint_dir.
            load_checkpoint_dir=None,  # If resuming, the directory to load checkpoints from
            wandb_enabled=True, # If true, will enable wandb logging.
            
            model_config=None,
            pytorch_weight_path="/media/sunhao/T71/Pi0.5/pi05_base_torch", # Optional path to a PyTorch checkpoint to load weights from, for finetuning
            freeze_stage="vlm",

            # Training hyperparameters.
            pytorch_training_precision="bfloat16", # Precision for PyTorch training.
            # The parameters below should be tuned based on tasks.
            # Refer to Pi0.5/OriginalCodes/openpi/src/openpi/training/config.py for some detailed examples
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            end_lr=2.5e-6,
            optimizer_b1=0.9,
            optimizer_b2=0.95,
            optimizer_eps=1e-8,
            optimizer_weight_decay=1e-10,
            optimizer_clip_gradient_norm=1.0,
            num_train_steps=30_000,

            # Data configuration.
            seed=42,
            batch_size=16*3, # Global batch size. Not per-GPU — the actual batch size per GPU will be calculated as `batch_size // GPUs`.
            num_workers=2,
            zero_grad_double_check=True,  # If true, will check that gradients are actually zeroed after optim.zero_grad() and log a warning if not.
            data_config=None,
        ),
    },
    "thermal_v4_original_trial": {
        "data_config": DataConfig(
            repo_id="thermtacvla/thermal_pi05_v1_v4",
            local_root="/home/ca/TTVLA/datasets/lerobot/thermal_pi05_v1_v4",
            asset_id="thermal_pi05_v1_v4_norm_stats",
            norm_stats_dir="/home/ca/TTVLA/assets/thermal_pi05_v1_v4_norm_stats",
            use_quantile_norm=True,
            video_backend="pyav",
            repack_transforms=(
                RepackTransform({
                    "images": {
                        "base_0_rgb": "observation.images.up",
                        "left_wrist_0_rgb": "observation.images.side",
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "task_index": "task_index",
                }),
            ),
            data_transforms=(ZeroRightWrist(),),
            action_sequence_keys=("action",),
            prompt_from_task=True,
            default_prompt=None,
            tokenizer_path="/home/ca/.cache/openpi/big_vision/paligemma_tokenizer.model",
        ),
        "model_config": Pi0Config(pi05=True),
        "train_config": TrainConfig(
            project_name="pi",
            exp_name="thermal_v4_original_trial",
            run_dir="/home/ca/TTVLA/checkpoints/pi05_v4_trial",
            log_interval=1,
            save_interval=5,
            resume=False,
            load_checkpoint_dir=None,
            wandb_enabled=False,
            model_config=None,
            pytorch_weight_path="/home/ca/ckpt",
            freeze_stage="vlm",
            pytorch_training_precision="bfloat16",
            warmup_steps=1,
            peak_lr=2.5e-5,
            decay_steps=5,
            end_lr=2.5e-6,
            optimizer_b1=0.9,
            optimizer_b2=0.95,
            optimizer_eps=1e-8,
            optimizer_weight_decay=1e-10,
            optimizer_clip_gradient_norm=1.0,
            num_train_steps=5,
            seed=42,
            batch_size=1,
            num_workers=0,
            zero_grad_double_check=True,
            data_config=None,
        ),
    },
    # ------------------------------------------------------------------
    # Fine-tune on datasets/raw_v4 (80 episodes, exported to LeRobot v3.0).
    # All paths are derived from this checkout, so the same entry works on
    # the cloud server once the files are placed as follows:
    #   dataset       <repo>/datasets/lerobot/thermal_pi05_v1_v4
    #   norm stats    <repo>/assets/thermal_pi05_v1_v4_norm_stats  (tracked)
    #   tokenizer     ~/.cache/openpi/big_vision/paligemma_tokenizer.model
    #   base weights  ~/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch
    #                 (output of convert/convert_jax_to_torch.py --pi05)
    # ------------------------------------------------------------------
    "thermal_v4_ft": {
        "data_config": DataConfig(
            repo_id="thermtacvla/thermal_pi05_v1_v4",
            local_root=str(_REPO_ROOT / "datasets/lerobot/thermal_pi05_v1_v4"),
            asset_id="thermal_pi05_v1_v4_norm_stats",
            norm_stats_dir=str(_REPO_ROOT / "assets/thermal_pi05_v1_v4_norm_stats"),
            use_quantile_norm=True,
            video_backend="pyav",
            repack_transforms=(
                RepackTransform({
                    "images": {
                        "base_0_rgb": "observation.images.up",
                        "left_wrist_0_rgb": "observation.images.side",
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "task_index": "task_index",
                }),
            ),
            data_transforms=(ZeroRightWrist(),),
            action_sequence_keys=("action",),
            prompt_from_task=True,
            default_prompt=None,
            tokenizer_path=str(pathlib.Path.home() / ".cache/openpi/big_vision/paligemma_tokenizer.model"),
        ),
        "model_config": Pi0Config(pi05=True),
        "train_config": TrainConfig(
            project_name="pi",
            exp_name="thermal_v4_ft",
            run_dir=str(_REPO_ROOT / "runs"),
            log_interval=10,
            # Each checkpoint is the full bf16 model (~6.6 GB); 5000 keeps
            # disk usage ~40 GB for the default 30k-step run.
            save_interval=5_000,
            resume=False,
            load_checkpoint_dir=None,
            wandb_enabled=True,
            model_config=None,
            pytorch_weight_path=str(
                pathlib.Path.home() / ".cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch"
            ),
            freeze_stage="vlm",
            pytorch_training_precision="bfloat16",
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            end_lr=2.5e-6,
            optimizer_b1=0.9,
            optimizer_b2=0.95,
            optimizer_eps=1e-8,
            optimizer_weight_decay=1e-10,
            optimizer_clip_gradient_norm=1.0,
            num_train_steps=30_000,
            seed=42,
            batch_size=32,
            num_workers=4,
            zero_grad_double_check=True,
            data_config=None,
        ),
    },
}

# Pure-PyTorch LoRA counterpart of ``thermal_v4_ft``.  Dense base weights are
# loaded once, while checkpoints contain adapters and the small action heads.
_thermal_v4_lora_base = ALLCONFIGS["thermal_v4_ft"]
ALLCONFIGS["thermal_v4_lora"] = {
    "data_config": _thermal_v4_lora_base["data_config"],
    "model_config": _thermal_v4_lora_base["model_config"],
    "train_config": dataclasses.replace(
        _thermal_v4_lora_base["train_config"],
        exp_name="thermal_v4_lora",
        freeze_stage=None,
        peak_lr=1.0e-4,
        end_lr=1.0e-5,
        lora=LoRATrainingConfig(
            enabled=True,
            attn_rank=16,
            ffn_rank=16,
            attn_alpha=16.0,
            ffn_alpha=16.0,
            dropout=0.05,
            apply_to="all",
            train_vision_encoder=False,
            train_non_lora_layers=True,
        ),
    ),
}

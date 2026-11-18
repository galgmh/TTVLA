"""
Usage
Single GPU:
  python finetune_new.py --config_name test_run
Multi-GPU (single node):
  torchrun --standalone --nnodes=1 --nproc_per_node=3 finetune_new.py --config_name test_run
"""
# Patches MUST be installed before any code touches `transformers`.
from pi05_model._patches import install as install_transformers_patches
install_transformers_patches()

import dataclasses
import gc
import logging
import os
import pathlib
import time

from typing import Literal

import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm
import wandb
import tyro

from pi05_model.config import Pi0Config
from pi05_model.model import PI0Pytorch
from pi05_model.observation import Observation
from data.data_loader import DataConfig, create_data_loader


@dataclasses.dataclass(frozen=False)
class TrainConfig:
    # Logging and checkpointing
    project_name: str = "pi" # For wandb logging
    exp_name: str = "debug"
    run_dir: str = "./runs"
    @property
    def checkpoint_dir(self) -> pathlib.Path:
        return (pathlib.Path(self.run_dir) / self.exp_name / "checkpoints").resolve()
    @property
    def log_dir(self) -> pathlib.Path:
        return (pathlib.Path(self.run_dir) / self.exp_name / "logs").resolve()
    log_interval: int = 100 # in steps
    save_interval: int = 1000 # in steps
    resume: bool = False # If true, will resume training from load_checkpoint_dir.
    load_checkpoint_dir: str = None  # If resuming, the directory to load checkpoints from
    wandb_enabled: bool = True # If true, will enable wandb logging.
    
    # Model configuration.
    model_config: Pi0Config = None
    pytorch_weight_path: str = None # Optional path to a PyTorch checkpoint to load weights from, for finetuning
    freeze_stage: str = None #  vlm, vlm_action_expert

    # Training hyperparameters.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16" # Precision for PyTorch training.
    # The parameters below should be tuned based on tasks.
    # Refer to Pi0.5/OriginalCodes/openpi/src/openpi/training/config.py for some detailed examples
    warmup_steps: int = 1_000
    peak_lr: float = 2.5e-5
    decay_steps: int = 30_000
    end_lr: float = 2.5e-6
    optimizer_b1 : float = 0.9
    optimizer_b2 : float = 0.95
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_clip_gradient_norm: float | None = 1.0
    num_train_steps: int = 30_000

    # Data configuration.
    seed: int = 42
    batch_size: int = 16 # Global batch size. Not per-GPU — the actual batch size per GPU will be calculated as `batch_size // GPUs`.
    num_workers: int = 2
    zero_grad_double_check: bool = True  # If true, will check that gradients are actually zeroed after optim.zero_grad() and log a warning if not.
    data_config: DataConfig = None


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def init_wandb(config: TrainConfig, resuming: bool, enabled: bool = True):
    """Initialize wandb logging."""
    if not enabled:
        wandb.init(mode="disabled")
        return

    log_dir = config.log_dir
    if not log_dir.exists():
        raise FileNotFoundError(f"Log directory {log_dir} does not exist.")

    if resuming:
        run_id = (log_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(name=config.exp_name, config=dataclasses.asdict(config), project=config.project_name)
        (log_dir / "wandb_id.txt").write_text(wandb.run.id)


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")

        # Set up debugging environment variables for DDP issues
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device


def cleanup_ddp():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


def build_datasets(config: TrainConfig):
    data_loader = create_data_loader(
        data_config=config.data_config,
        model_config=config.model_config,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
        shuffle=True,
        skip_norm_stats=False,
    )
    return data_loader
    

def get_model_state_dict(model):
    """Get state dict from model, handling DDP wrapper."""
    return (
        model.module.state_dict()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.state_dict()
    )


def get_model_parameters(model):
    """Get parameters from model, handling DDP wrapper."""
    return (
        model.module.parameters()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.parameters()
    )


def save_checkpoint(model, optimizer, global_step, config: TrainConfig, is_main, data_config: DataConfig):
    """Save a checkpoint with model state, optimizer state, and metadata."""
    if not is_main:
        return

    # Only save if it's time to save or if it's the final step
    if (global_step % config.save_interval == 0 and global_step > 0) or global_step == config.num_train_steps - 1:
        # Create temporary directory for atomic checkpoint saving
        ckpt_dir = config.checkpoint_dir / f"{global_step}"
        
        # Save model state using safetensors
        model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        safetensors.torch.save_model(model_to_save, ckpt_dir / "model.safetensors")
        
        # Save optimizer state using PyTorch format
        torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")        
        
        # Save training metadata
        metadata = {
            "global_step": global_step,
            "config": dataclasses.asdict(config),
            "timestamp": time.time(),
        }
        torch.save(metadata, ckpt_dir / "metadata.pt")
                
        logging.info(f"Saved checkpoint at step {global_step}.")

        # Log checkpoint to wandb
        if config.wandb_enabled:
            wandb.log({"checkpoint_step": global_step}, step=global_step)


def load_checkpoint(model, optimizer, load_checkpoint_dir, device):
    """
    Load the latest checkpoint and return the global step.
    Examples:
      /save/path/exp_name/checkpoints/20000/
    """

    checkpoint_dir = pathlib.Path(load_checkpoint_dir)

    if checkpoint_dir.name.isdigit():
        global_step = int(checkpoint_dir.name)
        ckpt_dir = checkpoint_dir
    else:
        raise ValueError(f"Checkpoint directory name must be a number representing the global step, got: {checkpoint_dir.name}")

    # Clear memory before loading checkpoints
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, global_step, "before_loading_checkpoint")

    try:
        # Load model state with error handling
        logging.info("Loading model state...")
        safetensors_path = ckpt_dir / "model.safetensors"

        if safetensors_path.exists():
            model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            safetensors.torch.load_model(model_to_load, safetensors_path, device=str(device))
            logging.info("Loaded model state from safetensors format")
        else:
            raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, global_step, "after_loading_model")

        # Load optimizer state with error handling
        logging.info("Loading optimizer state...")
        optimizer_path = ckpt_dir / "optimizer.pt"

        if optimizer_path.exists():
            optimizer_state_dict = torch.load(optimizer_path, map_location=device, weights_only=False)
            logging.info("Loaded optimizer state from pt format")
        else:
            raise FileNotFoundError(f"No optimizer checkpoint found at {ckpt_dir}")

        optimizer.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, global_step, "after_loading_optimizer")

        # Load metadata
        logging.info("Loading metadata...")
        metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
        global_step = metadata.get("global_step", global_step)
        del metadata
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, global_step, "after_loading_metadata")

        logging.info(f"Successfully loaded all checkpoint components from step {global_step}")
        return global_step

    except RuntimeError as e:
        if "out of memory" in str(e):
            # Clear memory and provide detailed error message
            torch.cuda.empty_cache()
            gc.collect()
            logging.error(f"Out of memory error while loading checkpoint: {e!s}")
            log_memory_usage(device, global_step, "after_oom_error")
            raise RuntimeError("Out of memory while loading checkpoint. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True") from e
        raise


def log_memory_usage(device, step, phase="unknown"):
    """Log detailed memory usage information."""
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    memory_free = memory_free / 1e9

    # Get more detailed memory info
    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

    # Get DDP info if available
    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}"
    )


def move_observation(obs: Observation, device: torch.device) -> Observation:
    return Observation(
        images={k: v.to(device, non_blocking=True) for k, v in obs.images.items()},
        image_masks={k: v.to(device, non_blocking=True) for k, v in obs.image_masks.items()},
        state=obs.state.to(device, non_blocking=True),
        tokenized_prompt=obs.tokenized_prompt.to(device, non_blocking=True),
        tokenized_prompt_mask=obs.tokenized_prompt_mask.to(device, non_blocking=True),
        token_ar_mask=None if obs.token_ar_mask is None else obs.token_ar_mask.to(device, non_blocking=True),
        token_loss_mask=None if obs.token_loss_mask is None else obs.token_loss_mask.to(device, non_blocking=True),
    )


def freeze_model(model, config: TrainConfig):
    """Set requires_grad on the raw PI0Pytorch model. Call this BEFORE DDP wrap
    so the Reducer captures only the truly trainable parameters."""
    if config.freeze_stage == "vlm":
        logging.info("Freezing VLM parameters for finetuning")
        model.paligemma_with_expert.paligemma.requires_grad_(False)
        model.paligemma_with_expert.gemma_expert.requires_grad_(True)
        model.action_in_proj.requires_grad_(True)
        model.action_out_proj.requires_grad_(True)
        if config.model_config.pi05:
            model.time_mlp_in.requires_grad_(True)
            model.time_mlp_out.requires_grad_(True)
        else:
            model.state_proj.requires_grad_(True)
            model.action_time_mlp_in.requires_grad_(True)
            model.action_time_mlp_out.requires_grad_(True)

    elif config.freeze_stage == "vlm_action_expert":
        logging.info("Freezing VLM and action expert parameters for finetuning")
        model.paligemma_with_expert.paligemma.requires_grad_(False)
        model.paligemma_with_expert.gemma_expert.requires_grad_(False)
        model.action_in_proj.requires_grad_(True)
        model.action_out_proj.requires_grad_(True)
        if config.model_config.pi05:
            model.time_mlp_in.requires_grad_(True)
            model.time_mlp_out.requires_grad_(True)
        else:
            model.state_proj.requires_grad_(True)
            model.action_time_mlp_in.requires_grad_(True)
            model.action_time_mlp_out.requires_grad_(True)
    else:
        raise ValueError(f"Invalid freeze_stage: {config.freeze_stage}. Must be one of: vlm, vlm_action_expert")


def train_loop(config: TrainConfig):
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(config.seed, local_rank)

    # Initialize checkpoint directory and wandb
    # Create checkpoint directory with experiment name
    if not config.resume:
        # For new runs, create experiment-specific checkpoint directory
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        config.log_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created experiment checkpoint directory: {config.checkpoint_dir}")
        logging.info(f"Created experiment log directory: {config.log_dir}")
    else:
        # For resume, checkpoint_dir is already set to the experiment directory
        logging.info(f"Using existing experiment checkpoint directory: {config.checkpoint_dir}")
        logging.info(f"Using existing experiment log directory: {config.log_dir}")

    # Initialize wandb (only on main process)
    if is_main:
        init_wandb(config, resuming=config.resume, enabled=config.wandb_enabled)

    # Build data loader using the unified data loader
    # Calculate effective batch size per GPU for DDP
    # For N GPUs, each GPU should get batch_size/N samples, so total across all GPUs is batch_size
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    effective_batch_size = config.batch_size // world_size
    logging.info(f"Using batch size per GPU: {effective_batch_size} (total batch size across {world_size} GPUs: {config.batch_size})")

    # Pass the original batch size to data loader - it will handle DDP splitting internally
    loader = build_datasets(config)

    # Build model directly on GPU with random weights — keep this fast and
    # symmetric across ranks so the DDP collective right after has minimal skew.
    object.__setattr__(config.model_config, "dtype", config.pytorch_training_precision)
    model = PI0Pytorch(config.model_config).to(device)

    # Freeze BEFORE DDP wrap so the Reducer only tracks truly trainable params.
    freeze_model(model, config)

    if hasattr(model, "gradient_checkpointing_enable"):
        enable_gradient_checkpointing = True
        model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing for memory optimization")
    else:
        enable_gradient_checkpointing = False
        logging.info("Gradient checkpointing is not supported for this model")

    # Log initial memory usage after model creation
    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

    # Enable memory optimizations for large-scale training
    if world_size >= 8:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set memory allocation configuration
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        logging.info("Enabled memory optimizations for 8+ GPU training")

    logging.info("Starting DDP setup (if applicable)")
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=False,   # Disable for memory efficiency
            gradient_as_bucket_view=True,  # Enable for memory efficiency
            static_graph=world_size >= 8,  # Enable for 8+ GPUs
        )
    logging.info("End DDP setup (if applicable)")

    if config.pytorch_weight_path is not None:
        logging.info(f"Loading weights from: {config.pytorch_weight_path}")
        model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        safetensors.torch.load_model(
            model_to_load,
            os.path.join(config.pytorch_weight_path, "model.safetensors"),
            device=str(device),
        )
        logging.info(f"Loaded PyTorch weights from {config.pytorch_weight_path}")

    # Log sample images to wandb on first batch, to help verify data loading and for visual reference.
    # Run this AFTER DDP wrap so it doesn't make rank 1 wait at the param-verify collective.
    if is_main and config.wandb_enabled and not config.resume:
        sample_data_loader = create_data_loader(
            data_config=config.data_config,
            model_config=config.model_config,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            seed=config.seed,
            shuffle=False,
            skip_norm_stats=False,
        )
        sample_batch = next(iter(sample_data_loader))
        observation, actions = sample_batch

        images_to_log = []
        images_dict = observation.images
        sample_bs = next(iter(images_dict.values())).shape[0]
        for i in range(min(5, sample_bs)):
            # CHW -> HWC for wandb.
            img_concatenated = torch.cat([img[i].permute(1, 2, 0) for img in images_dict.values()], axis=1)
            img_concatenated = img_concatenated.cpu().numpy()
            images_to_log.append(wandb.Image(img_concatenated))

        wandb.log({"camera_views": images_to_log}, step=0)
        logging.info("Logged sample images to wandb. Refer to W&B dashboard to verify data loading and for visual reference.")

        del sample_batch, observation, actions, images_to_log, img_concatenated, images_dict
        del sample_data_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Cleared sample batch and data loader from memory")

    # Create optimizer with config parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if is_main:
        n_trainable = sum(p.numel() for p in trainable_params)
        n_total = sum(p.numel() for p in model.parameters())
        logging.info(f"Trainable params: {n_trainable/1e6:.2f}M / {n_total/1e6:.2f}M")

    optim = torch.optim.AdamW(
        trainable_params,
        lr=config.peak_lr,
        betas=(config.optimizer_b1, config.optimizer_b2),
        eps=config.optimizer_eps,
        weight_decay=config.optimizer_weight_decay,
    )
    
    # Load checkpoint if resuming
    global_step = 0
    if config.resume:
        global_step = load_checkpoint(model, optim, config.load_checkpoint_dir, device)
        logging.info(f"Resumed training from step {global_step}")

    def lr_schedule(step: int):
        if step < config.warmup_steps:
            # Match JAX behavior: start from peak_lr / (warmup_steps + 1)
            init_lr = config.peak_lr / (config.warmup_steps + 1)
            return init_lr + (config.peak_lr - init_lr) * step / config.warmup_steps
        # cosine decay
        progress = min(1.0, (step - config.warmup_steps) / max(1, config.decay_steps - config.warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return config.end_lr + (config.peak_lr - config.end_lr) * cos

    model.train()
    start_time = time.time()
    infos = []  # Collect stats over log interval
    if is_main:
        logging.info(
            f"Running on: world_size={torch.distributed.get_world_size() if use_ddp else 1}"
        )
        logging.info(
            f"Training config: batch_size={config.batch_size}, effective_batch_size={effective_batch_size}, num_train_steps={config.num_train_steps}"
        )
        logging.info(f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}")
        logging.info(
            f"LR schedule: warmup={config.warmup_steps}, peak_lr={config.peak_lr:.2e}, decay_steps={config.decay_steps}, end_lr={config.end_lr:.2e}"
        )
        logging.info(
            f"Optimizer: {type(optim).__name__}, weight_decay={config.optimizer_weight_decay}, clip_norm={config.optimizer_clip_gradient_norm}"
        )
        logging.info(f"Training precision: {config.pytorch_training_precision}")

    # Training loop - iterate until we reach num_train_steps
    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main)
        if is_main else None
    )

    while global_step < config.num_train_steps:
        # Set epoch for distributed training
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(global_step // len(loader))

        for observation, actions in loader:
            if global_step >= config.num_train_steps:
                break

            # The unified data loader returns (observation, actions) tuple
            observation = move_observation(observation, device)
            actions = actions.to(torch.float32)
            actions = actions.to(device)

            # Update LR
            for pg in optim.param_groups:
                pg["lr"] = lr_schedule(global_step)

            losses = model(observation, actions)
            loss = losses.mean()
            loss.backward()

            # Log memory usage after backward pass
            if global_step < 5 and is_main and torch.cuda.is_available():
                log_memory_usage(device, global_step, "after_backward")

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optimizer_clip_gradient_norm)

            # Optimizer step
            optim.step()
            optim.zero_grad(set_to_none=True)

            # Clear gradients more aggressively
            if config.zero_grad_double_check:
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.detach_()
                        param.grad = None

            # Collect stats
            if is_main:
                infos.append(
                    {
                        "loss": loss.item(),
                        "learning_rate": optim.param_groups[0]["lr"],
                        "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    }
                )

            if is_main and (global_step % config.log_interval == 0):
                elapsed = time.time() - start_time

                # Average stats over log interval
                avg_loss = sum(info["loss"] for info in infos) / len(infos)
                avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)
                avg_grad_norm = sum(info["grad_norm"] for info in infos) / len(infos)

                logging.info(f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} grad_norm={avg_grad_norm:.2f} time={elapsed:.1f}s")

                # Log to wandb
                if config.wandb_enabled and len(infos) > 0:
                    log_payload = {
                        "loss": avg_loss,
                        "learning_rate": avg_lr,
                        "step": global_step,
                        "time_per_step": elapsed / config.log_interval,
                        "grad_norm": avg_grad_norm
                    }
                    wandb.log(log_payload, step=global_step)

                start_time = time.time()
                infos = []  # Reset stats collection

            global_step += 1
            # Save checkpoint using the new mechanism
            save_checkpoint(model, optim, global_step, config, is_main, config.data_config)

            # Update progress bar
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{optim.param_groups[0]['lr']:.2e}", "step": global_step})

    # Close progress bar
    if pbar is not None:
        pbar.close()

    # Finish wandb run
    if is_main and config.wandb_enabled:
        wandb.finish()

    cleanup_ddp()


def main(config_name: str) -> None:
    init_logging()
    from configs.configs import ALLCONFIGS

    if config_name not in ALLCONFIGS:
        raise ValueError(
            f"Unknown config name {config_name!r}. "
            f"Available: {sorted(ALLCONFIGS)}"
        )

    bundle = ALLCONFIGS[config_name]
    train_config: TrainConfig = bundle["train_config"]
    train_config.data_config = bundle["data_config"]
    train_config.model_config = bundle["model_config"]

    logging.info(f"Loaded config {config_name!r} from configs.configs.ALLCONFIGS")
    train_loop(train_config)


if __name__ == "__main__":
    tyro.cli(main)

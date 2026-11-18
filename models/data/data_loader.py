"""PyTorch-only data loader for Pi0/Pi0.5 fine-tuning.

Pipeline
--------
LeRobotDataset
   └─ _LeRobotPi05Dataset.__getitem__:
        torch sample → numpy → repack → data_transforms → Normalize
                    → NormalizeImages → ResizeImages → TokenizePrompt
                    → PadStatesAndActions → numpy sample
   └─ DataLoader collate (np.stack)
   └─ Pi05DataLoader.__iter__: numpy → torch → (Observation, actions)
"""

from __future__ import annotations

import dataclasses
import logging
import multiprocessing
import pathlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data
import lerobot.datasets.lerobot_dataset as lerobot_dataset

from pi05_model.config import Pi0Config
from pi05_model.observation import Observation
from pi05_model.preprocessing import IMAGE_RESOLUTION, resize_with_pad_torch
from utils.normalize import NormStats, load as load_norm_stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transforms (single-sample, numpy in / numpy out)
# ---------------------------------------------------------------------------
DataDict = dict[str, Any]
TransformFn = Callable[[DataDict], DataDict]


def compose(transforms: Sequence[TransformFn]) -> TransformFn:
    def _apply(data: DataDict) -> DataDict:
        for t in transforms:
            data = t(data)
        return data

    return _apply


def _flatten(d: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, Mapping):
            out.update(_flatten(v, prefix=f"{key}/"))
        else:
            out[key] = v
    return out


@dataclasses.dataclass(frozen=True)
class RepackTransform:
    """Rename / restructure raw dataset keys into the standard schema.

    `structure` is a (possibly nested) dict whose leaves are flat-key paths
    into the source sample. Use '/' to traverse nested fields.

    Example::

        RepackTransform({
            "images": {
                "base_0_rgb":   "observation.images.top",
                "left_wrist_0_rgb": "observation.images.left_wrist",
            },
            "state":   "observation.state",
            "actions": "actions",
            "prompt":  "prompt",  # optional
        })
    """

    structure: Mapping[str, Any]

    def __call__(self, data: DataDict) -> DataDict:
        flat = _flatten(data)

        def walk(node: Any) -> Any:
            if isinstance(node, Mapping):
                return {k: walk(v) for k, v in node.items()}
            if not isinstance(node, str):
                raise TypeError(f"RepackTransform leaves must be str, got {type(node)}")
            if node not in flat:
                raise KeyError(f"Missing key {node!r} in sample. Available: {sorted(flat)}")
            return flat[node]

        return walk(self.structure)


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask:
    """Resolve `task_index` (int) into a prompt string via the dataset's task table."""

    tasks: Mapping[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('PromptFromLeRobotTask requires "task_index" in the sample')
        idx = int(np.asarray(data["task_index"]).item())
        prompt = self.tasks.get(idx)
        if prompt is None:
            raise ValueError(f"task_index={idx} not found in task table {dict(self.tasks)}")
        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt:
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            return {**data, "prompt": self.prompt}
        return data


@dataclasses.dataclass(frozen=True)
class Normalize:
    """Apply z-score or quantile normalization per key listed in `norm_stats`.

    Only keys present in both `data` and `norm_stats` are touched. Image keys
    are intentionally not in norm_stats (images keep their [-1, 1] / [0, 255]
    range from the dataset).
    """

    norm_stats: Mapping[str, NormStats] | None
    use_quantiles: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data
        out = dict(data)
        for key, stats in self.norm_stats.items():
            if key not in out:
                continue
            x = np.asarray(out[key], dtype=np.float32)
            if self.use_quantiles:
                if stats.q01 is None or stats.q99 is None:
                    raise ValueError(f"Quantile norm requested but stats for {key!r} have no q01/q99")
                q01 = stats.q01[..., : x.shape[-1]]
                q99 = stats.q99[..., : x.shape[-1]]
                out[key] = (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
            else:
                mean = stats.mean[..., : x.shape[-1]]
                std = stats.std[..., : x.shape[-1]]
                out[key] = (x - mean) / (std + 1e-6)
        return out


@dataclasses.dataclass(frozen=True)
class NormalizeImages:
    """Map raw LeRobot image arrays to float32 in [-1, 1].

    LeRobot decodes video frames as float32 in [0, 1] and image-file features
    as uint8 in [0, 255]. The downstream PaliGemma / SigLIP vision tower (and
    `resize_with_pad_torch`'s float branch + `_augment_train`) all assume
    [-1, 1]. This transform unifies the convention before resizing/padding,
    so that `pad_value=-1.0` actually corresponds to black.
    """

    def __call__(self, data: DataDict) -> DataDict:
        if "images" not in data:
            return data
        out_images: dict[str, np.ndarray] = {}
        for k, v in data["images"].items():
            arr = np.asarray(v)
            if arr.dtype == np.uint8:
                arr = arr.astype(np.float32) / 255.0 * 2.0 - 1.0
            elif np.issubdtype(arr.dtype, np.floating):
                # LeRobot float frames live in [0, 1]; remap to [-1, 1].
                arr = arr.astype(np.float32) * 2.0 - 1.0
            else:
                raise ValueError(f"NormalizeImages: unsupported dtype for {k!r}: {arr.dtype}")
            out_images[k] = arr
        return {**data, "images": out_images}


@dataclasses.dataclass(frozen=True)
class ResizeImages:
    """Resize `images` dict to (height, width), aspect-preserving with padding.

    Reuses `pi05_model.preprocessing.resize_with_pad_torch` by going
    numpy → torch → numpy per image. Workers each have their own torch
    context, so this is safe and parallelizable.
    """

    height: int = IMAGE_RESOLUTION[0]
    width: int = IMAGE_RESOLUTION[1]

    def __call__(self, data: DataDict) -> DataDict:
        if "images" not in data:
            return data
        out_images: dict[str, np.ndarray] = {}
        for k, v in data["images"].items():
            arr = np.asarray(v)
            # Already at target size in HW positions (works for HWC and CHW with C in {1,3,4}).
            channels_last = arr.shape[-1] <= 4
            hw = arr.shape[-3:-1] if channels_last else arr.shape[-2:]
            if tuple(hw) == (self.height, self.width):
                out_images[k] = arr
                continue
            t = torch.from_numpy(np.ascontiguousarray(arr))
            t = resize_with_pad_torch(t, self.height, self.width)
            if t.dim() == 4 and arr.ndim == 3:
                t = t.squeeze(0)
            out_images[k] = t.numpy()
        return {**data, "images": out_images}


@dataclasses.dataclass(frozen=True)
class PadStatesAndActions:
    """Zero-pad state / actions on the trailing axis up to `model_action_dim`."""

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        out = dict(data)
        if "state" in out:
            out["state"] = _pad_to_dim(np.asarray(out["state"], dtype=np.float32), self.model_action_dim)
        if "actions" in out:
            out["actions"] = _pad_to_dim(np.asarray(out["actions"], dtype=np.float32), self.model_action_dim)
        return out


def _pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    cur = x.shape[axis]
    if cur >= target_dim:
        return x
    pad_widths = [(0, 0)] * x.ndim
    pad_widths[axis] = (0, target_dim - cur)
    return np.pad(x, pad_widths, mode="constant", constant_values=value)


# ---------------------------------------------------------------------------
# PaliGemma tokenizer (pi0.5: discrete state injected into the prompt string).
# This will right-pad the tokenized prompt to `model_config.max_token_len` and produce an attention mask.
# ---------------------------------------------------------------------------
class PaligemmaTokenizer:
    """Local SentencePiece wrapper. Provide a path to `paligemma_tokenizer.model`.

    The pi0.5 path (`state` is not None) bins the state into 256 buckets and
    encodes it as `Task: <prompt>, State: <bins>;\\nAction: ` before BOS.
    """

    def __init__(self, model_path: str | pathlib.Path, max_len: int = 200):
        import sentencepiece

        path = pathlib.Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"PaliGemma tokenizer model not found at {path}")
        with path.open("rb") as f:
            self._sp = sentencepiece.SentencePieceProcessor(model_proto=f.read())
        self._max_len = int(max_len)

    def tokenize(self, prompt: str, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            bins = np.linspace(-1.0, 1.0, 256 + 1)[:-1]
            disc = np.digitize(state, bins=bins) - 1
            state_str = " ".join(map(str, disc.tolist()))
            full = f"Task: {cleaned}, State: {state_str};\nAction: "
            tokens = self._sp.encode(full, add_bos=True)
        else:
            tokens = self._sp.encode(cleaned, add_bos=True) + self._sp.encode("\n")

        n = len(tokens)
        if n < self._max_len:
            pad = self._max_len - n
            mask = [True] * n + [False] * pad
            tokens = tokens + [0] * pad
        else:
            if n > self._max_len:
                logger.warning(
                    f"Token length ({n}) exceeds max_len ({self._max_len}); truncating. "
                    "Increase max_token_len in your model config if this is frequent."
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return np.asarray(tokens, dtype=np.int32), np.asarray(mask, dtype=bool)


@dataclasses.dataclass(frozen=True)
class TokenizePrompt:
    """Run the prompt + (optional) discrete state through the PaliGemma tokenizer."""

    tokenizer: PaligemmaTokenizer
    discrete_state_input: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        prompt = data.get("prompt")
        if prompt is None:
            raise ValueError(
                "TokenizePrompt requires a 'prompt' key (use InjectDefaultPrompt or PromptFromLeRobotTask)"
            )
        if not isinstance(prompt, str):
            prompt = str(np.asarray(prompt).item()) if hasattr(prompt, "item") else str(prompt)

        state = np.asarray(data["state"], dtype=np.float32) if self.discrete_state_input else None
        tokens, mask = self.tokenizer.tokenize(prompt, state=state)
        out = {k: v for k, v in data.items() if k != "prompt"}
        out["tokenized_prompt"] = tokens
        out["tokenized_prompt_mask"] = mask
        return out


# ---------------------------------------------------------------------------
# DataConfig — per-dataset spec consumed by create_data_loader.
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class DataConfig:
    """Everything needed to build a data loader for one dataset.

    `repack_transforms` is the only place the user customizes how raw
    LeRobot keys map to the standard schema {images, state, actions, prompt,
    task_index}. `data_transforms` is for robot-specific tweaks (e.g. delta
    actions). The trailing model_transforms (resize / tokenize / pad) are
    appended automatically from the model config.
    """

    repo_id: str
    # Local directory containing the LeRobot dataset (data/, meta/, videos/). When set, the
    # loader reads directly from disk and never touches the Hub. Required for fully offline use.
    local_root: str | pathlib.Path | None = None
    # Opaque identifier used as the asset directory name for norm_stats. Defaults to repo_id.
    asset_id: str | None = None

    # Path to a directory containing `norm_stats.json` (written by utils.normalize.save).
    # If both `norm_stats` and `norm_stats_dir` are unset, normalization is skipped.
    norm_stats: Mapping[str, NormStats] | None = None
    norm_stats_dir: str | pathlib.Path | None = None
    use_quantile_norm: bool = False

    # Per-dataset transforms. Each entry is a callable dict→dict.
    repack_transforms: Sequence[TransformFn] = ()
    data_transforms: Sequence[TransformFn] = ()

    # Which sample keys should be expanded along the action_horizon dimension by LeRobot.
    action_sequence_keys: Sequence[str] = ("actions",)
    # "pyav" avoids torchcodec binary/CUDA mismatches in the local environment.
    video_backend: str | None = None

    # If True, prompt is read from LeRobot's task table via task_index.
    prompt_from_task: bool = False
    # Fallback prompt when neither prompt_from_task nor a 'prompt' key is present.
    default_prompt: str | None = None

    # Required for TokenizePrompt — local path to paligemma_tokenizer.model.
    tokenizer_path: str | pathlib.Path | None = None

    def __post_init__(self) -> None:
        if self.asset_id is None:
            self.asset_id = self.repo_id


# ---------------------------------------------------------------------------
# Dataset wrapper.
# ---------------------------------------------------------------------------
class _LeRobotPi05Dataset(torch.utils.data.Dataset):
    """Thin wrapper that runs the transform pipeline on each sample.

    Stores transforms as a plain list (not a `compose()` closure) so the
    dataset can be pickled and shipped to spawn-mode DataLoader workers.
    """

    def __init__(
        self,
        lerobot_dataset: torch.utils.data.Dataset,
        transforms: Sequence[TransformFn],
    ) -> None:
        self._dataset = lerobot_dataset
        self._transforms = list(transforms)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> DataDict:
        raw = self._dataset[idx]
        sample = self._to_numpy_dict(raw) # convert torch tensors to numpy for the transform pipeline, which expects numpy in/out
        for t in self._transforms:
            sample = t(sample)
        return sample

    def _to_numpy_dict(self, obj: Any) -> Any:
        """Recursively convert torch tensors / scalars in a (possibly nested) dict to numpy."""
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy()
        if isinstance(obj, Mapping):
            return {k: self._to_numpy_dict(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], torch.Tensor):
            return np.stack([t.detach().cpu().numpy() for t in obj])
        return obj


# ---------------------------------------------------------------------------
# Collate + DataLoader wrapper.
# ---------------------------------------------------------------------------
def _collate(items: list[DataDict]) -> DataDict:
    """Stack a list of per-sample dicts into a batched dict (handles nested 'images')."""
    if not items:
        raise ValueError("Empty batch")
    first = items[0]
    out: dict[str, Any] = {}
    for key, ref in first.items():
        if isinstance(ref, Mapping):
            out[key] = {k: np.stack([item[key][k] for item in items], axis=0) for k in ref}
        elif isinstance(ref, np.ndarray):
            out[key] = np.stack([item[key] for item in items], axis=0)
        else:
            out[key] = np.asarray([item[key] for item in items])
    return out


def _worker_init_fn(worker_id: int) -> None:
    seed = (torch.initial_seed() + worker_id) & 0xFFFFFFFF
    np.random.seed(seed)


class Pi05DataLoader:
    """Yields (Observation, actions) tuples. Wraps torch.utils.data.DataLoader.

    Public API:
        - `__iter__()` → (Observation, actions)
        - `__len__()`  → number of batches per epoch
        - `set_epoch(epoch)` → forwarded to DistributedSampler if any
        - `data_config()` → original DataConfig (used by checkpointing code)
    """

    def __init__(
        self,
        torch_loader: torch.utils.data.DataLoader,
        sampler: torch.utils.data.Sampler | None,
        data_config: DataConfig,
    ) -> None:
        self._loader = torch_loader
        self._sampler = sampler
        self._data_config = data_config

    def data_config(self) -> DataConfig:
        return self._data_config

    def set_epoch(self, epoch: int) -> None:
        if isinstance(self._sampler, torch.utils.data.distributed.DistributedSampler):
            self._sampler.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self._loader)

    def __iter__(self):
        for batch in self._loader:
            yield _batch_to_observation(batch)


def _batch_to_observation(batch: DataDict) -> tuple[Observation, torch.Tensor]:
    images_np: dict[str, np.ndarray] = batch["images"]
    images = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in images_np.items()}

    if "image_masks" in batch:
        image_masks = {
            k: torch.from_numpy(np.ascontiguousarray(v)).to(torch.bool)
            for k, v in batch["image_masks"].items()
        }
    else:
        batch_size = next(iter(images.values())).shape[0]
        image_masks = {k: torch.ones(batch_size, dtype=torch.bool) for k in images}

    state = torch.from_numpy(np.ascontiguousarray(batch["state"])).to(torch.float32)
    tokens = torch.from_numpy(np.ascontiguousarray(batch["tokenized_prompt"])).to(torch.long)
    tokens_mask = torch.from_numpy(np.ascontiguousarray(batch["tokenized_prompt_mask"])).to(torch.bool)
    ar_mask = (
        torch.from_numpy(np.ascontiguousarray(batch["token_ar_mask"])).to(torch.long)
        if "token_ar_mask" in batch
        else None
    )
    loss_mask = (
        torch.from_numpy(np.ascontiguousarray(batch["token_loss_mask"])).to(torch.bool)
        if "token_loss_mask" in batch
        else None
    )
    actions = torch.from_numpy(np.ascontiguousarray(batch["actions"])).to(torch.float32)

    obs = Observation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=tokens,
        tokenized_prompt_mask=tokens_mask,
        token_ar_mask=ar_mask,
        token_loss_mask=loss_mask,
    )
    return obs, actions


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------
def create_data_loader(
    data_config: DataConfig,
    model_config: Pi0Config,
    batch_size: int,
    num_workers: int = 0,
    seed: int = 0,
    shuffle: bool = True,
    skip_norm_stats: bool = False,
) -> Pi05DataLoader:
    """
    Build the LeRobot → transforms → DataLoader → (Observation, actions) pipeline.
    `batch_size` is the GLOBAL batch size; under DDP it is divided across ranks.
    """

    if data_config.tokenizer_path is None:
        raise ValueError(
            "DataConfig.tokenizer_path is required (path to paligemma_tokenizer.model). "
            "See gs://big_vision/paligemma_tokenizer.model for the upstream file."
        )

    action_horizon = model_config.action_horizon
    assert action_horizon is not None  # filled in by Pi0Config.__post_init__

    local_root = pathlib.Path(data_config.local_root) if data_config.local_root is not None else None
    if local_root is not None and not local_root.is_dir():
        raise FileNotFoundError(
            f"DataConfig.local_root={local_root} does not exist or is not a directory. "
            "Expected a LeRobot dataset folder containing data/, meta/, videos/."
        )

    meta = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id, root=local_root)
    base = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        root=local_root,
        delta_timestamps={
            key: [t / meta.fps for t in range(action_horizon)]
            for key in data_config.action_sequence_keys
        }, # For keys in action_sequence_keys, expand each sample along the time dimension by repeating frames at intervals of 1/fps up to the action horizon. This allows training with multi-step action sequences.
        download_videos=local_root is None,
        video_backend=data_config.video_backend,
    )

    norm_stats = None if skip_norm_stats else data_config.norm_stats
    if not skip_norm_stats and norm_stats is None and data_config.norm_stats_dir is not None:
        norm_stats = load_norm_stats(pathlib.Path(data_config.norm_stats_dir))
    if not skip_norm_stats and norm_stats is None:
        logger.warning(
            "No norm_stats provided; running without normalization. "
            "Run a stats pass and set DataConfig.norm_stats / norm_stats_dir for proper training."
        )

    tokenizer = PaligemmaTokenizer(data_config.tokenizer_path, max_len=model_config.max_token_len)

    transforms: list[TransformFn] = [*data_config.repack_transforms]
    if data_config.prompt_from_task:
        # `meta.tasks` shape depends on LeRobot version:
        #   - older builds: dict-like {task_index: task_str}, usable as-is
        #   - v3.0+:        pandas DataFrame indexed by task_str with a `task_index` column,
        #                   needs inverting to {task_index: task_str}
        raw_tasks = meta.tasks
        if hasattr(raw_tasks, "iterrows"):  # pandas DataFrame
            tasks_map = {
                int(row["task_index"]): str(task_str)
                for task_str, row in raw_tasks.iterrows()
            }
        else:
            tasks_map = {int(k): str(v) for k, v in dict(raw_tasks).items()}
        transforms.append(PromptFromLeRobotTask(tasks_map))
    if data_config.default_prompt is not None:
        transforms.append(InjectDefaultPrompt(data_config.default_prompt))
    transforms.extend(data_config.data_transforms)
    transforms.append(Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm))
    transforms.append(NormalizeImages())
    transforms.append(ResizeImages(*IMAGE_RESOLUTION))
    transforms.append(TokenizePrompt(tokenizer, discrete_state_input=model_config.pi05))
    transforms.append(PadStatesAndActions(model_config.action_dim))

    dataset = _LeRobotPi05Dataset(base, transforms)

    sampler: torch.utils.data.Sampler | None = None
    local_batch_size = batch_size
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        if batch_size % world_size != 0:
            raise ValueError(f"batch_size={batch_size} not divisible by world_size={world_size}")
        local_batch_size = batch_size // world_size
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=dist.get_rank(),
            shuffle=shuffle,
            drop_last=True,
            seed=seed,
        )

    if local_batch_size > len(dataset):
        raise ValueError(
            f"local_batch_size ({local_batch_size}) > dataset size ({len(dataset)}). "
            "Lower batch_size or use a larger dataset."
        )

    generator = torch.Generator()
    generator.manual_seed(seed)

    mp_context = multiprocessing.get_context("spawn") if num_workers > 0 else None

    torch_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=_collate,
        worker_init_fn=_worker_init_fn,
        drop_last=True,
        generator=generator,
        multiprocessing_context=mp_context,
        persistent_workers=num_workers > 0,
        pin_memory=True,
    )

    logger.info(
        f"Built data loader: repo_id={data_config.repo_id} "
        f"len(dataset)={len(dataset)} batches/epoch={len(torch_loader)} "
        f"local_batch_size={local_batch_size} num_workers={num_workers}"
    )

    return Pi05DataLoader(torch_loader, sampler, data_config)


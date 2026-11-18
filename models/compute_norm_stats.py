"""Compute normalization statistics for a Pi0.5 LeRobot dataset.

Iterates the dataset with **only** the repack
(+ optional data) transforms applied — i.e. the same view of `state`
and `actions` that training will see *before* `PadStatesAndActions`.
Running mean/std/q01/q99 are accumulated and saved as
`<output_dir>/<asset_id>/norm_stats.json`, which is the layout
`utils.normalize.load(...)` and `DataConfig.norm_stats_dir` expect.

Example:
the file will be save to output_dir/asset_id/norm_stats.json:
python compute_norm_stats.py \
    --args.repo_id lerobot/svla_so101_pickplace+my_own_label \
    --args.local_root /media/sunhao/T71/svla_so101_pickplace \
    --args.asset_id svla_so101_pickplace_norm_stats+action_horizon_10 \
    --args.output_dir /media/sunhao/T71/Assets \
    --args.action-horizon 10
"""

from __future__ import annotations

import dataclasses
import logging
import multiprocessing
import pathlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.utils.data
import tqdm
import tyro
import lerobot.datasets.lerobot_dataset as lerobot_dataset

from data.data_loader import (
    DataConfig,
    RepackTransform,
    TransformFn,
)
from utils.normalize import RunningStats, save as save_norm_stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_numpy(obj: Any) -> Any:
    """Recursively convert torch tensors in a (possibly nested) dict to numpy."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    if isinstance(obj, Mapping):
        return {k: _to_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], torch.Tensor):
        return np.stack([t.detach().cpu().numpy() for t in obj])
    return obj


class _StatsDataset(torch.utils.data.Dataset):
    """LeRobot sample → numpy → repack/data_transforms only (no padding/tokenize).

    Holds transforms as a plain list (not a closure from `compose()`) so the
    dataset can be pickled and shipped to spawn-mode DataLoader workers.
    """

    def __init__(
        self,
        base: torch.utils.data.Dataset,
        transforms: Sequence[TransformFn],
    ) -> None:
        self._base = base
        self._transforms = list(transforms)

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = _to_numpy(self._base[idx])
        for t in self._transforms:
            sample = t(sample)
        return sample


class _CollateKeys:
    """Picklable replacement for `lambda items: _collate_keys(items, keys)`."""

    def __init__(self, keys: Sequence[str]) -> None:
        self.keys = tuple(keys)

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        return {k: np.stack([np.asarray(item[k]) for item in items], axis=0) for k in self.keys}


def _build_dataset(data_config: DataConfig, action_horizon: int) -> _StatsDataset:
    local_root = pathlib.Path(data_config.local_root) if data_config.local_root is not None else None
    if local_root is not None and not local_root.is_dir():
        raise FileNotFoundError(
            f"local_root={local_root} does not exist or is not a directory. "
            "Expected a LeRobot dataset folder containing data/, meta/, videos/."
        )

    meta = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id, root=local_root)
    base = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        root=local_root,
        delta_timestamps={
            key: [t / meta.fps for t in range(action_horizon)]
            for key in data_config.action_sequence_keys
        },
        download_videos=local_root is None,
        video_backend=data_config.video_backend,
    )

    # Same order as create_data_loader, minus prompt/model transforms.
    # Prompts are not needed for state/action stats, so we skip prompt_from_task
    # and InjectDefaultPrompt here.
    transforms: list[TransformFn] = [*data_config.repack_transforms, *data_config.data_transforms]
    return _StatsDataset(base, transforms)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Args:
    # Dataset identification. `repo_id` is a label only when `local_root` is set.
    repo_id: str = "lerobot/svla_so101_pickplace+my_own_label"
    local_root: str | None = "/media/sunhao/T71/svla_so101_pickplace"

    # Directory name used under `output_dir` for the saved norm_stats.json.
    # Must match the `asset_id` of the DataConfig used at training time.
    asset_id: str = "svla_so101_pickplace_my_own_label"

    # Where to write `<asset_id>/norm_stats.json`. Point training's
    # `DataConfig.norm_stats_dir` at the resulting subdirectory.
    output_dir: str = "./assets"

    # Should match `model_config.action_horizon` used at training time
    # (pi0.5 default is 10; pi0 default is 50).
    action_horizon: int = 10

    # Iteration controls. Match the loader's collate so batch sizes here
    # only affect speed, not the resulting stats.
    batch_size: int = 32
    num_workers: int = 4
    max_frames: int | None = None
    seed: int = 0


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Matches the repack used in TestData.py but drops `images` (not needed
    # for state/action norm_stats and would needlessly decode video frames).
    data_config = DataConfig(
        repo_id=args.repo_id,
        local_root=args.local_root,
        asset_id=args.asset_id,
        action_sequence_keys=("action",),
        video_backend="pyav",
        repack_transforms=(
            RepackTransform({
                "state": "observation.state",
                "actions": "action",
            }),
        ),
    )

    dataset = _build_dataset(data_config, args.action_horizon)
    keys = ["state", "actions"]
    logger.info(f"Dataset size: {len(dataset)} samples")

    if args.max_frames is not None and args.max_frames < len(dataset):
        num_batches = args.max_frames // args.batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // args.batch_size
        shuffle = False
    if num_batches < 1:
        raise ValueError(
            f"num_batches={num_batches}: batch_size ({args.batch_size}) is larger than the "
            f"available frame budget ({args.max_frames if args.max_frames is not None else len(dataset)})."
        )

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    mp_context = multiprocessing.get_context("spawn") if args.num_workers > 0 else None

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=_CollateKeys(keys),
        drop_last=True,
        generator=generator,
        multiprocessing_context=mp_context,
        persistent_workers=args.num_workers > 0,
    )

    stats = {key: RunningStats() for key in keys}
    seen = 0
    for batch in tqdm.tqdm(loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key], dtype=np.float32))
        seen += 1
        if seen >= num_batches:
            break

    norm_stats = {key: stat.get_statistics() for key, stat in stats.items()}

    output_path = pathlib.Path(args.output_dir) / args.asset_id
    logger.info(f"Writing stats to: {output_path / 'norm_stats.json'}")
    save_norm_stats(output_path, norm_stats)

    for key, ns in norm_stats.items():
        logger.info(
            f"  {key}: shape={tuple(ns.mean.shape)} "
            f"mean[:3]={np.array2string(ns.mean[:3], precision=4)} "
            f"std[:3]={np.array2string(ns.std[:3], precision=4)}"
        )


if __name__ == "__main__":
    tyro.cli(main)

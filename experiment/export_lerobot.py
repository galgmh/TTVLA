#!/usr/bin/env python
"""Export a ThermTacVLA format-1.6 HDF5 file as a LeRobot v2.1 dataset.

The output is the layout openpi's LeRobot loader expects (one parquet per
episode + one MP4 per camera per episode), so a pi0.5 fine-tune only needs a
data config pointing at the directory. Nothing here imports Genesis: h5py,
numpy, Pillow, imageio and pyarrow are enough.

    python experiment/export_lerobot.py DATASET.hdf5 --out datasets/lerobot/pick_cold_can_p16

Contract (see experiment/common.py for the exact index layout):

    observation.state   16 dims (proprio16), 28 (touch28) or 30 (full30)
    action              7 dims, LIBERO delta-pose + gripper synergy
    observation.images.image         agentview_rgb  (H,W,3) uint8
    observation.images.wrist_image   eye_in_hand_rgb
    per episode         language_instruction attr -> meta/tasks.jsonl

The pure-vision baseline uses ``--state-mode proprio16`` (the default). The
thermal rungs use ``touch28`` / ``full30``; those names mirror
``sim/configs/.../pick_cold_can.yaml``'s measured fingertip channel and never
read the scene's ground-truth object temperature.

Splits are contiguous over the exported episodes (train, then val, then test);
they are written to ``meta/info.json`` under ``splits``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiment.common import (  # noqa: E402
    ACTION_NAMES,
    ACTION_DIM,
    IMAGE_KEYS,
    assemble_state_arrays,
    state_dim,
    state_names,
)

# (LeRobot feature key, HDF5 obs channel). The names match openpi's LIBERO data
# config (`LeRobotLiberoDataConfig`), which is the closest starting point.
CAMERAS: tuple[tuple[str, str], ...] = (
    (IMAGE_KEYS[0], "agentview_rgb"),
    (IMAGE_KEYS[1], "eye_in_hand_rgb"),
)

CODEBASE_VERSION = "v2.1"
CHUNKS_SIZE = 1000


# --------------------------------------------------------------------- stats


class RowStats:
    """Streaming mean/std/min/max/count for fixed-width rows."""

    def __init__(self, dim: int):
        self.dim = int(dim)
        self.n = 0
        self.s = np.zeros(self.dim, dtype=np.float64)
        self.ss = np.zeros(self.dim, dtype=np.float64)
        self.lo = np.full(self.dim, np.inf, dtype=np.float64)
        self.hi = np.full(self.dim, -np.inf, dtype=np.float64)

    def blank(self) -> "RowStats":
        return RowStats(self.dim)

    def update(self, rows: Any) -> None:
        arr = np.asarray(rows, dtype=np.float64).reshape(-1, self.dim)
        if arr.shape[0] == 0:
            return
        self.n += int(arr.shape[0])
        self.s += arr.sum(axis=0)
        self.ss += np.square(arr).sum(axis=0)
        self.lo = np.minimum(self.lo, arr.min(axis=0))
        self.hi = np.maximum(self.hi, arr.max(axis=0))

    def merge(self, other: "RowStats") -> None:
        if other.dim != self.dim:
            raise ValueError(f"stats dim mismatch: {self.dim} vs {other.dim}")
        self.n += other.n
        self.s += other.s
        self.ss += other.ss
        self.lo = np.minimum(self.lo, other.lo)
        self.hi = np.maximum(self.hi, other.hi)

    def to_dict(self) -> dict[str, Any]:
        if self.n == 0:
            raise ValueError("cannot serialise empty stats")
        mean = self.s / self.n
        var = np.maximum(self.ss / self.n - np.square(mean), 0.0)
        return {
            "mean": [float(v) for v in mean],
            "std": [float(v) for v in np.sqrt(var)],
            "min": [float(v) for v in self.lo],
            "max": [float(v) for v in self.hi],
            # LeRobot stores count as np.array([n]) and its loader rejects
            # anything with ndim == 0, so this must be a one-element list.
            "count": [int(self.n)],
        }


class ImageStats:
    """Streaming per-channel stats over frames; ``count`` is the frame count."""

    def __init__(self) -> None:
        self.frames = 0
        self.pixels = 0
        self.s = np.zeros(3, dtype=np.float64)
        self.ss = np.zeros(3, dtype=np.float64)
        self.lo = np.full(3, np.inf, dtype=np.float64)
        self.hi = np.full(3, -np.inf, dtype=np.float64)

    def blank(self) -> "ImageStats":
        return ImageStats()

    def update(self, frame: np.ndarray) -> None:
        rows = np.asarray(frame, dtype=np.float64).reshape(-1, 3)
        self.frames += 1
        self.pixels += int(rows.shape[0])
        self.s += rows.sum(axis=0)
        self.ss += np.square(rows).sum(axis=0)
        self.lo = np.minimum(self.lo, rows.min(axis=0))
        self.hi = np.maximum(self.hi, rows.max(axis=0))

    def merge(self, other: "ImageStats") -> None:
        self.frames += other.frames
        self.pixels += other.pixels
        self.s += other.s
        self.ss += other.ss
        self.lo = np.minimum(self.lo, other.lo)
        self.hi = np.maximum(self.hi, other.hi)

    def to_dict(self) -> dict[str, Any]:
        if self.pixels == 0:
            raise ValueError("cannot serialise empty image stats")
        mean = self.s / self.pixels
        var = np.maximum(self.ss / self.pixels - np.square(mean), 0.0)
        # LeRobot stores image stats with shape (3, 1, 1) (one value per
        # channel plus two singleton axes) and rejects a flat (3,) array.
        return {
            "mean": [[[float(v)]] for v in mean],
            "std": [[[float(v)]] for v in np.sqrt(var)],
            "min": [[[float(v)]] for v in self.lo],
            "max": [[[float(v)]] for v in self.hi],
            "count": [int(self.frames)],
        }


# ------------------------------------------------------------------- helpers


def _need(name: str, pip_name: str | None = None):
    try:
        return __import__(name)
    except ImportError as exc:  # pragma: no cover - environment dependent
        hint = f" (pip install {pip_name or name})" if pip_name else ""
        raise SystemExit(f"ERROR: {name} is required{hint}") from exc


def _decode_attr(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _attr_str(attrs: Any, key: str) -> str:
    if key not in attrs:
        raise KeyError(f"demo has no {key!r} attr; was the file written by scripts/generate.py?")
    return _decode_attr(attrs[key])


def _read_fps(f: h5py.File) -> float:
    raw = f.attrs.get("env_args")
    if raw is None:
        return 20.0
    try:
        env = json.loads(_decode_attr(raw))
        return float(env["sim"]["control_hz"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return 20.0


def _video_info(size: int, fps: float) -> dict[str, Any]:
    return {
        "video.fps": float(fps),
        "video.height": int(size),
        "video.width": int(size),
        "video.channels": 3,
        "video.codec": "h264",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "has_audio": False,
    }


def _feature_float(dim: int, names: list[str] | None) -> dict[str, Any]:
    return {"dtype": "float32", "shape": [int(dim)], "names": names}


def _feature_video(size: int, fps: float) -> dict[str, Any]:
    info = _video_info(size, fps)
    return {
        "dtype": "video",
        "shape": [int(size), int(size), 3],
        "names": ["height", "width", "channels"],
        "info": info,
    }


def _resize(frame: np.ndarray, size: int) -> np.ndarray:
    if size == frame.shape[0] and size == frame.shape[1]:
        return np.asarray(frame, dtype=np.uint8)
    from PIL import Image

    return np.asarray(Image.fromarray(frame).resize((size, size), Image.BILINEAR))


def _write_video(
    path: Path,
    frames_handle: Any,
    *,
    length: int,
    size: int,
    fps: float,
    stats: ImageStats,
) -> None:
    imageio = _need("imageio", "imageio imageio-ffmpeg")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(path),
        fps=float(fps),
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
    )
    try:
        for t in range(length):
            frame = _resize(np.asarray(frames_handle[t]), size)
            stats.update(frame)
            writer.append_data(frame)
    finally:
        writer.close()


def _write_parquet(
    path: Path,
    *,
    state: np.ndarray,
    action: np.ndarray,
    timestamp: np.ndarray,
    frame_index: np.ndarray,
    episode_index: int,
    index: np.ndarray,
    task_index: int,
) -> None:
    _need("pyarrow", "pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq

    length = int(state.shape[0])
    table = pa.table(
        {
            "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32())),
            "action": pa.array(action.tolist(), type=pa.list_(pa.float32())),
            "timestamp": pa.array(np.asarray(timestamp, dtype=np.float32)),
            "frame_index": pa.array(np.asarray(frame_index, dtype=np.int64)),
            "episode_index": pa.array(np.full(length, int(episode_index), dtype=np.int64)),
            "index": pa.array(np.asarray(index, dtype=np.int64)),
            "task_index": pa.array(np.full(length, int(task_index), dtype=np.int64)),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path), compression="snappy")


def _split_ranges(n: int, val_frac: float, test_frac: float) -> dict[str, str]:
    if n <= 0:
        return {}
    n_test = int(round(n * max(0.0, test_frac)))
    n_val = int(round(n * max(0.0, val_frac)))
    while n_val + n_test > n:
        if n_test >= n_val and n_test > 0:
            n_test -= 1
        elif n_val > 0:
            n_val -= 1
        else:
            break
    n_train = n - n_val - n_test
    out: dict[str, str] = {}
    if n_train:
        out["train"] = f"0:{n_train}"
    if n_val:
        out["val"] = f"{n_train}:{n_train + n_val}"
    if n_test:
        out["test"] = f"{n_train + n_val}:{n}"
    return out


# --------------------------------------------------------------------- export


def export(args: argparse.Namespace) -> int:
    src = Path(args.hdf5).expanduser()
    if not src.is_file():
        print(f"ERROR: {src} not found")
        return 2
    out = Path(args.out).expanduser()
    if out.resolve() == REPO.resolve():
        print("ERROR: refusing to export over the repository root; pass a dataset directory")
        return 2
    if out.exists() and not out.is_dir():
        print(f"ERROR: {out} exists and is not a directory")
        return 2
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        print(f"ERROR: {out} exists and is not empty; pass --overwrite to replace it")
        return 2
    if out.exists() and args.overwrite:
        # Only the directories this exporter owns -- never the parent itself
        # (a mistyped --out must not take datasets/ with it).
        for name in ("data", "videos", "meta"):
            sub = out / name
            if sub.exists():
                shutil.rmtree(sub)
    out.mkdir(parents=True, exist_ok=True)

    _need("pyarrow", "pyarrow")
    _need("PIL", "Pillow")
    _need("imageio", "imageio imageio-ffmpeg")

    with h5py.File(src, "r") as f:
        if "data" not in f:
            print(f"ERROR: {src} has no 'data' group (not a tempvla file?)")
            return 2
        format_version = _decode_attr(f.attrs.get("format_version", "unknown"))
        fps = float(args.fps) if args.fps is not None else _read_fps(f)

        demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        plans: list[tuple[str, str, int]] = []
        for demo in demos:
            g = f["data"][demo]
            success = bool(g.attrs.get("success", False))
            if not success and not args.include_failed:
                continue
            instruction = _attr_str(g.attrs, "language_instruction")
            length = int(g["actions"].shape[0])
            if length <= 0:
                continue
            plans.append((demo, instruction, length))
            if args.limit is not None and len(plans) >= args.limit:
                break

        if not plans:
            print("ERROR: no exportable episodes (all failed? try --include-failed)")
            return 2

        # Tasks are deduplicated in first-appearance order.
        task_index: dict[str, int] = {}
        for _, instruction, _ in plans:
            task_index.setdefault(instruction, len(task_index))

        # One image size for the whole dataset.
        first_obs = f["data"][plans[0][0]]["obs"]
        present = [(key, channel) for key, channel in CAMERAS if channel in first_obs]
        if not present:
            print("ERROR: the file has no RGB channels; regenerate with cameras enabled")
            return 2
        native = int(first_obs[present[0][1]].shape[2])
        size = int(args.image_size) if args.image_size is not None else native
        if size % 2:
            print(f"ERROR: --image-size must be even for yuv420p (got {size})")
            return 2

        print(
            f"export: {src.name}  format {format_version}  "
            f"{len(plans)} episodes  fps {fps:g}  {size}x{size}  state {args.state_mode} "
            f"({state_dim(args.state_mode)}D)"
        )
        print(f"        cameras: {', '.join(k for k, _ in present)}")

        splits = _split_ranges(len(plans), args.val_frac, args.test_frac)
        global_stats: dict[str, RowStats | ImageStats] = {}

        def merge_global(key: str, stats: RowStats | ImageStats) -> None:
            if key not in global_stats:
                global_stats[key] = stats.blank()
            global_stats[key].merge(stats)  # type: ignore[arg-type]

        episodes_meta: list[dict[str, Any]] = []
        episodes_stats: list[dict[str, Any]] = []
        index0 = 0

        for episode_index, (demo, instruction, length) in enumerate(plans):
            g = f["data"][demo]
            obs = g["obs"]
            for _key, channel in present:
                if channel not in obs:
                    print(f"ERROR: {demo} has no obs/{channel}; every episode must match")
                    return 2

            state = assemble_state_arrays(
                obs["ee_pos"][:],
                obs["ee_ori"][:],
                obs["gripper_states"][:],
                obs["joint_states"][:],
                fingertip_grid=(
                    obs["temperature/fingertip_grid"][:]
                    if args.state_mode == "full30"
                    else None
                ),
                tactile_force=(
                    obs["tactile/force"][:]
                    if args.state_mode in ("touch28", "full30")
                    else None
                ),
                mode=args.state_mode,
            )
            action = np.asarray(g["actions"][:], dtype=np.float32)
            if action.shape[0] != state.shape[0]:
                print(
                    f"ERROR: {demo}: actions {action.shape[0]} != observations "
                    f"{state.shape[0]}"
                )
                return 2
            if action.shape[1] < ACTION_DIM:
                print(f"ERROR: {demo}: actions have {action.shape[1]} dims, need {ACTION_DIM}")
                return 2
            action = action[:, :ACTION_DIM]

            chunk = episode_index // CHUNKS_SIZE
            stem = f"episode_{episode_index:06d}"
            parquet_path = out / "data" / f"chunk-{chunk:03d}" / f"{stem}.parquet"

            frame_index = np.arange(length, dtype=np.int64)
            timestamp = frame_index.astype(np.float64) / float(fps)
            index = np.arange(index0, index0 + length, dtype=np.int64)
            task_idx = int(task_index[instruction])

            _write_parquet(
                parquet_path,
                state=state,
                action=action,
                timestamp=timestamp,
                frame_index=frame_index,
                episode_index=episode_index,
                index=index,
                task_index=task_idx,
            )

            ep_stats: dict[str, RowStats | ImageStats] = {
                "observation.state": RowStats(state.shape[1]),
                "action": RowStats(ACTION_DIM),
                "timestamp": RowStats(1),
                "frame_index": RowStats(1),
                "episode_index": RowStats(1),
                "index": RowStats(1),
                "task_index": RowStats(1),
            }
            ep_stats["observation.state"].update(state)  # type: ignore[union-attr]
            ep_stats["action"].update(action)  # type: ignore[union-attr]
            ep_stats["timestamp"].update(timestamp[:, None])  # type: ignore[union-attr]
            ep_stats["frame_index"].update(frame_index[:, None])  # type: ignore[union-attr]
            ep_stats["episode_index"].update(
                np.full((length, 1), episode_index, dtype=np.float64)  # type: ignore[union-attr]
            )
            ep_stats["index"].update(index[:, None])  # type: ignore[union-attr]
            ep_stats["task_index"].update(
                np.full((length, 1), task_idx, dtype=np.float64)  # type: ignore[union-attr]
            )

            for key, channel in present:
                rel = (out / "videos" / f"chunk-{chunk:03d}" / key / f"{stem}.mp4")
                cam_stats = ImageStats()
                _write_video(
                    rel,
                    obs[channel],
                    length=length,
                    size=size,
                    fps=fps,
                    stats=cam_stats,
                )
                ep_stats[key] = cam_stats

            for key, stats in ep_stats.items():
                merge_global(key, stats)

            episodes_meta.append(
                {"episode_index": episode_index, "tasks": [instruction], "length": int(length)}
            )
            episodes_stats.append(
                {
                    "episode_index": episode_index,
                    "stats": {key: stats.to_dict() for key, stats in ep_stats.items()},
                }
            )
            print(
                f"  [{episode_index + 1}/{len(plans)}] {demo}  T={length:4d}  "
                f"{instruction}"
            )
            index0 += length

    # ------------------------------------------------------------------ meta
    meta = out / "meta"
    meta.mkdir(parents=True, exist_ok=True)

    features: dict[str, Any] = {
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "observation.state": _feature_float(state_dim(args.state_mode), state_names(args.state_mode)),
        "action": _feature_float(ACTION_DIM, ACTION_NAMES),
    }
    for key, _channel in present:
        features[key] = _feature_video(size, fps)

    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": "panda",
        "total_episodes": len(plans),
        "total_frames": int(index0),
        "total_tasks": len(task_index),
        "total_videos": len(plans) * len(present),
        "total_chunks": (len(plans) + CHUNKS_SIZE - 1) // CHUNKS_SIZE,
        "chunks_size": CHUNKS_SIZE,
        "fps": float(fps),
        "splits": splits,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    (meta / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")

    with (meta / "tasks.jsonl").open("w", encoding="utf-8") as fh:
        for text, idx in task_index.items():
            fh.write(json.dumps({"task_index": idx, "task": text}) + "\n")

    with (meta / "episodes.jsonl").open("w", encoding="utf-8") as fh:
        for row in episodes_meta:
            fh.write(json.dumps(row) + "\n")

    with (meta / "episodes_stats.jsonl").open("w", encoding="utf-8") as fh:
        for row in episodes_stats:
            fh.write(json.dumps(row) + "\n")

    stats_payload = {
        key: stats.to_dict()
        for key, stats in global_stats.items()
        if not (isinstance(stats, RowStats) and stats.n == 0)
    }
    (meta / "stats.json").write_text(
        json.dumps(stats_payload, indent=2) + "\n", encoding="utf-8"
    )

    print()
    print(f"wrote {out}")
    print(f"  episodes {len(plans)}  frames {index0}  tasks {len(task_index)}")
    print(f"  splits   {splits}")
    print(f"  state    {args.state_mode} ({state_dim(args.state_mode)}D)")
    print()
    print("next: python experiment/check_lerobot.py " + str(out))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("hdf5", type=Path, help="format-1.6 file written by scripts/generate.py")
    ap.add_argument("--out", type=Path, required=True, help="LeRobot dataset directory")
    ap.add_argument(
        "--state-mode",
        choices=("proprio16", "touch28", "full30"),
        default="proprio16",
        help="policy state width (default proprio16 = the pure-vision baseline)",
    )
    ap.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="square video size; 0 keeps the recorded resolution (default 256)",
    )
    ap.add_argument("--fps", type=float, default=None, help="default: env_args.sim.control_hz")
    ap.add_argument("--val-frac", type=float, default=0.08, help="episode fraction (default 0.08)")
    ap.add_argument("--test-frac", type=float, default=0.04, help="episode fraction (default 0.04)")
    ap.add_argument("--limit", type=int, default=None, help="export at most N episodes")
    ap.add_argument(
        "--include-failed",
        action="store_true",
        help="also export failed episodes (default: successful only)",
    )
    ap.add_argument("--overwrite", action="store_true", help="replace a non-empty --out")
    args = ap.parse_args()
    if args.image_size == 0:
        args.image_size = None
    return export(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""Convert thermal-pi05-v1 raw HDF5 episodes to a LeRobot dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np

from sim.recording import read_episode
from sim.rotations import quat_error_rotvec

STATE_NAMES = [
    "joint_0",
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "gripper",
]
ACTION_NAMES = ["dx", "dy", "dz", "drot_x", "drot_y", "drot_z", "gripper"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert thermal-pi05-v1 HDF5 to LeRobot")
    parser.add_argument("--source", default="datasets/thermal_pi05_v1/raw")
    parser.add_argument("--root", default="datasets/lerobot/thermal_pi05_v1")
    parser.add_argument("--repo-id", default="thermtacvla/thermal_pi05_v1")
    parser.add_argument("--robot-type", default="panda")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--include-failed", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _episode_action(state: np.ndarray, target_pose: np.ndarray, gripper: float) -> np.ndarray:
    delta_xyz = target_pose[:3] - state[:3]
    delta_rot = quat_error_rotvec(target_pose[3:], state[3:])
    return np.concatenate([delta_xyz, delta_rot, [gripper]]).astype(np.float32)


def _lerobot_class():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        try:
            from lerobot.datasets import LeRobotDataset
        except ImportError as exc:
            raise RuntimeError(
                "A LeRobot installation containing lerobot.datasets is required. "
                "Install the LeRobot version used by models."
            ) from exc
    return LeRobotDataset


def _features(height: int, width: int) -> dict:
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": {"axes": STATE_NAMES},
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": {"axes": ACTION_NAMES},
        },
        "observation.images.up": {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.side": {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        },
    }


def main() -> int:
    args = build_parser().parse_args()
    source = Path(args.source)
    root = Path(args.root)
    files = sorted(source.rglob("*.hdf5"))
    if not files:
        raise SystemExit(f"no HDF5 files under {source}")
    if args.overwrite and root.exists():
        shutil.rmtree(root)
    if root.exists() and any(root.iterdir()):
        raise SystemExit(f"output exists: {root}; use --overwrite")

    LeRobotDataset = _lerobot_class()
    selected = 0
    skipped = 0
    dataset = None

    for path in files:
        if args.limit is not None and selected >= args.limit:
            break

        arrays, metadata, _ = read_episode(path)
        meta = dict(metadata)
        success = bool(meta.get("success", False))
        if not args.include_failed and not success:
            skipped += 1
            continue

        target_pose = np.asarray(arrays["control/eef_target"], dtype=np.float64)
        measured_pose = np.asarray(arrays["robot/eef_pose"], dtype=np.float64)
        gripper_target = np.asarray(arrays["control/gripper_target"], dtype=np.float64)
        joint_position = np.asarray(arrays["robot/joint_position"], dtype=np.float32)
        gripper_width = np.asarray(arrays["robot/gripper_width"], dtype=np.float32)
        base_images = np.asarray(arrays["camera/base_0_rgb"], dtype=np.uint8)
        wrist_images = np.asarray(arrays["camera/wrist_0_rgb"], dtype=np.uint8)
        frames = min(
            len(target_pose),
            len(measured_pose),
            len(gripper_target),
            len(joint_position),
            len(gripper_width),
            len(base_images),
            len(wrist_images),
        )
        if frames < 2:
            skipped += 1
            continue

        if dataset is None:
            height, width = base_images.shape[1:3]
            dataset = LeRobotDataset.create(
                repo_id=args.repo_id,
                root=root,
                fps=int(args.fps),
                robot_type=str(args.robot_type),
                features=_features(height, width),
            )

        prompt = str(meta["instruction"])
        for frame_index in range(frames):
            state8 = np.concatenate(
                [joint_position[frame_index, :7], [gripper_width[frame_index]]]
            ).astype(np.float32)
            action7 = _episode_action(
                measured_pose[frame_index],
                target_pose[frame_index],
                float(gripper_target[frame_index]),
            )
            dataset.add_frame({
                "observation.state": state8,
                "action": action7,
                "observation.images.up": np.ascontiguousarray(base_images[frame_index]),
                "observation.images.side": np.ascontiguousarray(wrist_images[frame_index]),
                "task": prompt,
            })
        dataset.save_episode()
        selected += 1
        print(f"[{selected}] {path.name} frames={frames}")

    if dataset is None:
        raise SystemExit("no successful episodes selected")
    dataset.finalize()
    print(f"完成：写入 {selected} 集，跳过 {skipped} 集 -> {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

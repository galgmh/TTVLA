"""从单个 episode HDF5 导出 base/wrist RGB MP4。

用法::

    python datasets/export_hdf5_mp4.py episode.hdf5 --mode all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

CAMERA_KEYS = ("base_0_rgb", "wrist_0_rgb")


def _fps_from_control_time(time: np.ndarray) -> float:
    time = np.asarray(time, dtype=np.float64).reshape(-1)
    if time.size < 2:
        return 20.0
    dt = float(np.median(np.diff(time)))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"control/time 无法推导帧率：dt={dt}")
    return 1.0 / dt


def _write_mp4(path: Path, frames: np.ndarray, fps: float) -> None:
    import cv2

    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.shape[0] == 0:
        raise ValueError(f"RGB 帧形状应为 [T,H,W,3]，实际为 {frames.shape}")
    height, width = (int(v) for v in frames.shape[1:3])
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"无法创建 MP4：{path}")
    try:
        for frame in frames:
            rgb = np.asarray(frame, dtype=np.uint8)
            writer.write(np.ascontiguousarray(rgb[..., ::-1]))  # RGB -> BGR
    finally:
        writer.release()


def _side_by_side(base: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    """Align two RGB streams and concatenate them horizontally."""
    import cv2

    count = min(len(base), len(wrist))
    if count == 0:
        raise ValueError("相机序列不能为空")
    base = np.asarray(base[:count], dtype=np.uint8)
    wrist = np.asarray(wrist[:count], dtype=np.uint8)
    target_h = int(base.shape[1])
    if wrist.shape[1] != target_h:
        target_w = round(wrist.shape[2] * target_h / wrist.shape[1])
        wrist = np.stack([
            cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            for frame in wrist
        ])
    return np.concatenate([base, wrist], axis=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="从 HDF5 导出双相机 RGB MP4")
    parser.add_argument("episode", type=Path, help="输入 episode HDF5")
    parser.add_argument("--out-dir", type=Path, default=None, help="输出目录（默认与 HDF5 同级）")
    parser.add_argument("--fps", type=float, default=None, help="覆盖帧率（默认从 control/time 推导）")
    parser.add_argument(
        "--mode", choices=("separate", "combined", "all"), default="all",
        help="分别导出、左右拼接，或两者都导出（默认 all）",
    )
    args = parser.parse_args()

    import h5py

    if not args.episode.is_file():
        raise FileNotFoundError(args.episode)
    out_dir = args.out_dir or args.episode.parent
    with h5py.File(args.episode, "r") as handle:
        fps = float(args.fps) if args.fps is not None else _fps_from_control_time(handle["control/time"][...])
        if fps <= 0.0:
            raise ValueError("--fps 必须大于 0")
        frames = {}
        for key in CAMERA_KEYS:
            dataset = f"camera/{key}"
            if dataset not in handle:
                raise KeyError(f"HDF5 缺少 {dataset}")
            frames[key] = handle[dataset][...]

        outputs = []
        if args.mode in ("separate", "all"):
            for key in CAMERA_KEYS:
                output = out_dir / f"{args.episode.stem}_{key}.mp4"
                _write_mp4(output, frames[key], fps)
                outputs.append(output)
        if args.mode in ("combined", "all"):
            output = out_dir / f"{args.episode.stem}_combined.mp4"
            _write_mp4(
                output,
                _side_by_side(frames["base_0_rgb"], frames["wrist_0_rgb"]),
                fps,
            )
            outputs.append(output)

    print(f"帧率: {fps:.3f} fps")
    for output in outputs:
        print(f"已写出: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
